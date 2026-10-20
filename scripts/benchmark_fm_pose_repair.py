"""Autonomous two-checkpoint pose repair with three-stage physical validity.

GPU workers share only the explicitly selected device. Each sample's input is
prepared once and shared across models. Native pose is evaluator-only after
corruption. CPU case evaluators have bounded runtimes and save partial progress.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dirmo.chemistry.graphs import graph_check, graph_id, mapped, read_sdf, smiles, stereo_from_coordinates, write_sdf
from dirmo.evaluation.pocket import prepare_fixed_pocket, protein_atoms, structure
from dirmo.gpu import inventory, resolve_device, worker_environment
from dirmo.io import digest, sha256_file
from probe_fm_pose_batch import CHECKPOINTS, distort, torsion_region, monitor_memory


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def emit(**fields):
    print(json.dumps(fields), flush=True)


def prepare(args):
    root = args.output.resolve()
    plan = read(root / "study.json")
    if plan:
        if not args.resume or plan['cases_requested'] != args.cases or plan['seeds'] != args.seeds:
            raise ValueError("Study exists; use --resume with original cases/seeds")
        return plan
    from Bio.SeqUtils import seq1
    roots=[args.dataset,*args.extra_dataset]
    source_cases=[]
    seen=set()
    for source_root in roots:
        for directory in sorted(source_root.iterdir()):
            if directory.is_dir() and directory.name not in seen:
                source_cases.append(directory)
                seen.add(directory.name)
    np.random.default_rng(2042).shuffle(source_cases)
    old = read(ROOT / "runs/fm_quality_gpu3_20261013_v1/study.json", {"cases": []})
    old_ids = {c['case_id'] for c in old['cases']}
    exclusions, cases = [], []
    proteins, scaffolds = set(), set()
    for directory in source_cases:
        case_id = directory.name
        try:
            if case_id in old_ids:
                raise ValueError("Excluded earlier 20-case pilot")
            protein = directory / f"{case_id}_protein.pdb"
            ligand = directory / f"{case_id}_ligand.sdf"
            original = read_sdf(ligand)
            options = Chem.RemoveHsParameters()
            options.removeWithQuery = True
            original = mapped(Chem.RemoveHs(original, options))
            if not 10 <= original.GetNumAtoms() <= 64:
                raise ValueError("Heavy atom eligibility [10,64]")
            if any(a.GetAtomicNum() not in {6, 7, 8, 9, 15, 16, 17, 35, 53} for a in original.GetAtoms()):
                raise ValueError("Unsupported ligand element in this study")
            clean = Chem.MolFromSmiles(smiles(original, True))
            index = {a.GetAtomMapNum(): a.GetIdx() for a in clean.GetAtoms()}
            clean = Chem.RenumberAtoms(clean, [index[a.GetAtomMapNum()] for a in original.GetAtoms()])
            conformer = Chem.Conformer(clean.GetNumAtoms())
            conformer.Set3D(True)
            for i, xyz in enumerate(original.GetConformer().GetPositions()):
                conformer.SetAtomPosition(i, xyz)
            clean.AddConformer(conformer)
            clean = stereo_from_coordinates(clean)
            if args.coordinate_stereo_baseline:
                # Missing source stereo may be inferred from the unmodified
                # experimental XYZ. Explicit source stereo must still agree.
                matches=clean.GetSubstructMatches(original,useChirality=True,uniquify=False,maxMatches=4096)
                if tuple(range(clean.GetNumAtoms())) not in matches:
                    raise ValueError("Explicit source stereo contradicts native coordinates")
            elif smiles(clean) != smiles(original):
                raise ValueError("Native coordinate stereo disagrees with source graph")
            region = torsion_region(clean)
            pdb = structure(str(protein))
            sequence = tuple(sorted("".join(seq1(r.resname) for r in chain if r.id[0] == ' ')
                                    for chain in next(pdb.get_models())))
            protein_id = digest(sequence)
            if protein_id in proteins and not args.allow_related_complexes:
                raise ValueError("Duplicate exact protein sequence")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles(clean)))
            if scaffold and scaffold in scaffolds and not args.allow_related_complexes:
                raise ValueError("Duplicate exact Murcko scaffold")
            case_root = root / "cases" / case_id
            native = case_root / "native.sdf"
            write_sdf(native, clean)
            persisted = read_sdf(native)
            if not graph_check(clean, persisted, True)['matches']:
                raise ValueError("Native SDF stereo roundtrip failure")
            pocket = case_root / "pocket.pdb"
            prep = prepare_fixed_pocket(str(protein), str(native), str(pocket), 6.)
            positions, _, _ = protein_atoms(str(pocket))
            if not 20 <= len(positions) <= 1000:
                raise ValueError("Pocket eligibility [20,1000] heavy atoms")
            case = {"case_id": case_id, "target_id": protein_id, "scaffold": scaffold,
                    "source_dataset":str(directory.parent),
                    "source_stereo_smiles":smiles(original),
                    "coordinate_stereo_baseline":args.coordinate_stereo_baseline,
                    "native_sdf": str(native), "protein_path": str(protein), "pocket_path": str(pocket),
                    "atoms": clean.GetNumAtoms(), "pocket_atoms": len(positions), "region": region,
                    "graph_smiles": smiles(clean, True), "graph_id": graph_id(clean),
                    "source_hashes": {"protein": sha256_file(protein), "ligand": sha256_file(ligand)},
                    "pocket_preparation": prep, "inputs": []}
            for level, difficulty in enumerate(('small', 'large')):
                for offset in range(args.seeds):
                    seed = 4242 + level * 10000 + offset
                    rng = np.random.default_rng(seed)
                    angle = float(rng.uniform(20,40) if level == 0 else rng.uniform(50,70))
                    angle *= int(rng.choice([-1,1]))
                    file = case_root / "inputs" / difficulty / f"{seed}.sdf"
                    write_sdf(file, stereo_from_coordinates(distort(persisted, region, angle)))
                    damaged = read_sdf(file)
                    if not graph_check(persisted, damaged, True)['matches']:
                        raise ValueError("Corruption/serialization changed graph stereo")
                    held = [i for i in range(clean.GetNumAtoms()) if i not in region['free_indices']]
                    drift = np.linalg.norm(persisted.GetConformer().GetPositions()[held] -
                                           damaged.GetConformer().GetPositions()[held], axis=1).max()
                    if drift > .001:
                        raise ValueError("Corruption moved a held atom")
                    case['inputs'].append({"case_id":case_id,"input_sdf":str(file),
                          "seed":seed,"difficulty":difficulty,"angle_degrees":angle,
                          "free_maps":region['free_maps'],"pocket_path":str(pocket)})
            save(case_root / "case.json", case)
            cases.append(case)
            proteins.add(protein_id)
            if scaffold:
                scaffolds.add(scaffold)
            emit(status="prepared", cases=len(cases), target=args.cases, case=case_id,
                 atoms=case['atoms'],pocket_atoms=case['pocket_atoms'])
            save(root / "preparation.json", {"prepared":len(cases),"requested":args.cases,"exclusions":len(exclusions)})
            if len(cases)==args.cases:
                break
        except Exception as exc:
            exclusions.append({"case_id":case_id,"reason":f"{type(exc).__name__}: {exc}"})
    save(root / "eligibility-exclusions.json", exclusions)
    if len(cases) < min(30,args.cases):
        raise ValueError(f"Only {len(cases)} suitable cases; full experiment not started")
    plan = {"cases_requested":args.cases,"actual_cases":len(cases),"cases":cases,"seeds":args.seeds,
            "steps":args.steps,"batch_cap":args.batch_size,"physical_gpu":args.gpu,
            "planned_outputs":len(cases)*args.seeds*2*2,"gpu_workers":2,
            "selection":("Seed-shuffled distinct complexes; exclude pilot; related targets/scaffolds retained"
                         if args.allow_related_complexes else "Seed-shuffled eligibility; exclude previous pilot; exact protein/scaffold dedup"),
            "related_complexes_retained":args.allow_related_complexes,
            "coordinate_stereo_baseline":args.coordinate_stereo_baseline,
            "unique_exact_proteins":len(proteins),"unique_nonempty_scaffolds":len(scaffolds),
            "source_datasets":[str(p) for p in roots],
            "region_selection":"Deterministic rotatable terminal region, 2..8 heavy atoms, independent of metrics",
            "corruption":"Signed torsion 20..40 degrees / 50..70 degrees; same input for both checkpoints",
            "conditions":"All graph features held; only coordinates outside the free mask held",
            "stereo_contract":"Derived from generated XYZ and checked after sampling; no separate stereo input feature",
            "pretraining_overlap":"unknown", "training":False,"llm_calls":0,
            "native_physical_validity_selection":False,
            "source_revision":"739a33c2aab74d074dd24605b9a2e34dc809ae9c",
            "script_sha256":sha256_file(__file__),
            "backend_sha256":sha256_file(ROOT/'src/dirmo/generation/pose_repair.py')}
    save(root / "study.json",plan)
    return plan


def generate(args):
    import threading
    from dirmo.config import FMConfig
    from dirmo.generation.pose_repair import PoseRepairBackend
    root=args.output
    plan=read(root/'study.json')
    checkpoint=args.checkpoint
    filename,expected_hash=CHECKPOINTS[checkpoint]
    config=FMConfig(gpu=args.gpu,port=8876,python=sys.executable,repo_path='/home/uslee/code/flowr_root',
                    checkpoint_path=str(args.checkpoints/filename),checkpoint_sha256=expected_hash,
                    inpaint_mode='graph',max_atoms=64,max_pocket_atoms=1000)
    backend=PoseRepairBackend(config)
    (root/'phases'/checkpoint/'complete.json').unlink(missing_ok=True)
    save(root/'phases'/checkpoint/'health.json',{**backend.backend.health(),'pid':os.getpid(),'physical_gpu':args.gpu})
    cases=sorted(plan['cases'],key=lambda c:(c['pocket_atoms']+c['atoms'])**2,reverse=True)
    pending=[]
    for case in cases:
        pending.extend({**item,'output_dir':str(root/'draws'/checkpoint/case['case_id']/item['difficulty']/str(item['seed'])),
                        'shape_atoms':case['atoms']+case['pocket_atoms']}
                       for item in case['inputs'] if not
                       (root/'draws'/checkpoint/case['case_id']/item['difficulty']/str(item['seed'])/'result.json').exists())
    announced=set()
    cases_by_id={c['case_id']:c for c in cases}
    def mark_completed_cases(keys=None):
        for case in cases if keys is None else (cases_by_id[key] for key in keys):
            key=case['case_id']
            marker=root/'phases'/checkpoint/f'{key}.done.json'
            if key in announced:
                continue
            if all((root/'draws'/checkpoint/key/i['difficulty']/str(i['seed'])/'result.json').exists()
                   for i in case['inputs']):
                save(marker,{'case_id':key,'outputs':len(case['inputs']),'checkpoint':checkpoint})
                announced.add(key)
                emit(status='case_generation_complete',checkpoint=checkpoint,cases=len(announced),total=len(cases))
    mark_completed_cases()
    history=root/'phases'/checkpoint/'batches.jsonl'
    while pending:
        # One target per batch avoids unvalidated mixed-size padding effects.
        # The two workers target most of GPU 4's memory; OOM reduces the batch.
        current_case=pending[0]['case_id']
        same_case=[i for i in pending[:plan['batch_cap']] if i['case_id']==current_case]
        shape=(max(i['shape_atoms'] for i in same_case)/231.)**2
        per_sample=(.35 if checkpoint=='base' else .25)*1.4*shape
        budget=26. if checkpoint=='base' else 18.
        capacity=max(1,min(plan['batch_cap'],len(same_case),int(budget/per_sample)))
        batch_size=max(n for n in (1,2,4,8,16,32) if n<=capacity)
        while True:
            backend.torch.cuda.empty_cache()
            items=same_case[:batch_size]
            records,stopped=[],threading.Event()
            monitor=threading.Thread(target=monitor_memory,args=(os.environ['DIRMO_GPU_UUID'],stopped,records),daemon=True)
            monitor.start()
            try:
                results,stats=backend.generate(items,plan['steps'])
                pending=pending[len(items):]
                values=[v for v in records if 'used_mib' in v]
                stats.update(checkpoint=checkpoint,case_ids=[i['case_id'] for i in items],
                             statuses=dict(Counter(x['status'] for x in results)),
                             peak_shared_device_used_mib=max((v['used_mib'] for v in values),default=None))
                with history.open('a') as f:
                    f.write(json.dumps(stats)+'\n')
                emit(status='batch_complete',checkpoint=checkpoint,cases=sorted({i['case_id'] for i in items}),
                     batch_size=len(items),remaining=len(pending),peak_allocated_gib=stats['peak_allocated_bytes']/2**30)
                break
            except backend.torch.OutOfMemoryError:
                stats={'status':'oom','checkpoint':checkpoint,'case_ids':[i['case_id'] for i in items],
                       'batch_size':len(items),'partial_forward_calls':getattr(backend,'last_forward_calls',None),
                       'partial_sample_nfe':getattr(backend,'last_forward_calls',0)*len(items),
                       'peak_allocated_bytes':backend.torch.cuda.max_memory_allocated()}
                with history.open('a') as f:
                    f.write(json.dumps(stats)+'\n')
                if batch_size==1:
                    for item in items:
                        save(Path(item['output_dir'])/'result.json',{'status':'oom','sample':item,'batch_stats':stats})
                    pending=pending[len(items):]
                    emit(**stats)
                    break
                else:
                    batch_size=max(1,batch_size//2)
                emit(**stats)
            except Exception as exc:
                traceback.print_exc()
                # Isolate a malformed target rather than fail its batch peers.
                if len(items)>1 and getattr(backend,'last_forward_calls',0)==0:
                    batch_size=max(1,len(items)//2)
                    emit(status='preprocessing_retry',checkpoint=checkpoint,batch_size=batch_size,reason=str(exc))
                    continue
                for item in items:
                    save(Path(item['output_dir'])/'result.json',{'status':'failed','sample':item,
                         'reason':f'{type(exc).__name__}: {exc}',
                         'partial_forward_calls':getattr(backend,'last_forward_calls',0)})
                pending=pending[len(items):]
                break
            finally:
                stopped.set()
                monitor.join(timeout=6)
        mark_completed_cases([current_case])
    save(root/'phases'/checkpoint/'complete.json',{'status':'complete','cases':len(cases)})


def physical_category(metrics,key='pocket'):
    if not metrics or metrics.get('build_failed'):
        return 'missing'
    pb=metrics.get(key,{})
    checks=pb.get('checks',{})
    if pb.get('passes'):
        return 'valid'
    if any(v is False for v in checks.values()):
        return 'invalid'
    return 'missing'


def evaluate_case(args):
    from dirmo.evaluation.metrics import posebusters
    from dirmo.evaluation.pose_repair_checks import configure_pose_repair_checks
    from posebusters import PoseBusters
    from posebusters.modules import energy_ratio
    from posebusters.tools.loading import safe_load_mol
    original=energy_ratio.get_average_energy
    @lru_cache(maxsize=None)
    def average(inchi,n_confs=50,num_threads=0):
        return original(inchi,n_confs,1)
    energy_ratio.get_average_energy=average
    configure_pose_repair_checks()
    case=read(args.output/'cases'/args.case_id/'case.json')
    file=args.output/'evaluations'/f'{args.case_id}.json'
    result=read(file,{'case_id':args.case_id,'inputs':{},'generated':{'base':{},'v2':{}},'relaxed':{}})
    plan=read(args.output/'study.json',{})
    if plan.get('baseline_output'):
        baseline_root=Path(plan['baseline_output'])
        baseline_case=read(baseline_root/'cases'/args.case_id/'case.json',{})
        baseline=read(baseline_root/'evaluations'/f'{args.case_id}.json',{})
        if baseline_case.get('native_sdf')==case['native_sdf'] and baseline.get('native'):
            result.setdefault('native',baseline['native'])
            for item in case['inputs']:
                key=f"{item['difficulty']}/{item['seed']}"
                previous=baseline.get('inputs',{}).get(key,{})
                if previous.get('sample',{}).get('input_sdf')==item['input_sdf']:
                    result['inputs'].setdefault(key,previous)
                    if key in baseline.get('relaxed',{}):
                        result['relaxed'].setdefault(key,baseline['relaxed'][key])
    native=read_sdf(case['native_sdf'])
    free=case['region']['free_indices']
    held=[i for i in range(native.GetNumAtoms()) if i not in free]
    loading=PoseBusters(config='dock').config.get('loading',{}).get('mol_cond',{})
    @lru_cache(maxsize=2)
    def conditioning_protein(path):
        # Parse the identical complete protein once, using upstream's exact
        # loading settings. No waters/cofactors/atoms or checks are removed.
        return safe_load_mol(Path(path),**loading)

    def measure(path):
        try:
            mol=stereo_from_coordinates(read_sdf(path))
            graph=graph_check(native,mol,True)
            m={'sdf':str(path),'graph_preserved':graph['matches']}
            if mol.GetNumAtoms()==native.GetNumAtoms():
                left,right=native.GetConformer().GetPositions(),mol.GetConformer().GetPositions()
                m['held_max_drift']=float(np.linalg.norm(left[held]-right[held],axis=1).max())
                m['free_rmsd_atom_map']=float(np.sqrt(np.mean(np.sum((left[free]-right[free])**2,axis=1))))
                if graph['matches']:
                    query,product=Chem.Mol(native),Chem.Mol(mol)
                    for molecule in (query,product):
                        for atom in molecule.GetAtoms():
                            atom.SetAtomMapNum(0)
                    matches=product.GetSubstructMatches(query,useChirality=True,uniquify=False,maxMatches=4096)
                    allowed=[match for match in matches if all(match[i]==i for i in held)]
                    m['free_rmsd']=min((float(np.sqrt(np.mean(np.sum((left[free]-right[[match[i] for i in free]])**2,axis=1))))
                                       for match in allowed),default=m['free_rmsd_atom_map'])
                    m['symmetry_matches']=len(allowed)
            for key,protein in (('pocket',case['pocket_path']),('full_protein',case['protein_path'])):
                try:
                    m[key]=posebusters(mol,protein,conditioning_mol=conditioning_protein(protein))
                except Exception as exc:
                    m[key]={'passes':False,'error':f'{type(exc).__name__}: {exc}'}
            return m
        except Exception as exc:
            return {'sdf':str(path),'build_failed':True,'error':f'{type(exc).__name__}: {exc}'}

    if 'native' not in result:
        result['native']=measure(case['native_sdf'])
        save(file,result)
    for item in case['inputs']:
        key=f"{item['difficulty']}/{item['seed']}"
        if key not in result['inputs']:
            result['inputs'][key]={'sample':item,'metrics':measure(item['input_sdf'])}
            save(file,result)
        if key not in result['relaxed']:
            relaxed=args.output/'cases'/args.case_id/'relaxed'/item['difficulty']/f"{item['seed']}.sdf"
            try:
                mol=Chem.AddHs(read_sdf(item['input_sdf']),addCoords=True)
                ff=AllChem.UFFGetMoleculeForceField(mol)
                if ff is None:
                    raise ValueError('UFF unavailable')
                for i in held:
                    ff.AddFixedPoint(i)
                converged=ff.Minimize(maxIts=200)
                write_sdf(relaxed,stereo_from_coordinates(Chem.RemoveHs(mol)))
                result['relaxed'][key]={'metrics':measure(relaxed),'uff_convergence_code':converged,
                     'definition':'Ligand-only UFF, held heavy coordinates fixed, no pocket energy term'}
            except Exception as exc:
                result['relaxed'][key]={'metrics':{'build_failed':True,'error':str(exc)}}
            save(file,result)
        for checkpoint in CHECKPOINTS:
            if key not in result['generated'][checkpoint]:
                directory=args.output/'draws'/checkpoint/args.case_id/item['difficulty']/str(item['seed'])
                generated=read(directory/'result.json',{'status':'missing'})
                metrics=measure(directory/'raw.sdf') if generated.get('sdf_path') else {'build_failed':True,'reason':generated.get('reason',generated['status'])}
                result['generated'][checkpoint][key]={'sample':item,'generation_status':generated['status'],'metrics':metrics}
                save(file,result)
    result['complete']=True
    result.pop('evaluation_incomplete',None)
    result.pop('reason',None)
    save(file,result)
    emit(status='case_evaluation_complete',case=args.case_id)


def aggregate(root):
    plan=read(root/'study.json')
    if not plan:
        return
    groups=defaultdict(list)
    natives=[]
    complete_cases=0
    incomplete_cases=[]
    for case in plan['cases']:
        data=read(root/'evaluations'/f"{case['case_id']}.json",{})
        complete_cases+=bool(data.get('complete'))
        if data.get('evaluation_incomplete'):
            incomplete_cases.append(case['case_id'])
        if 'native' in data:
            natives.append(data['native'])
        for key,item in data.get('inputs',{}).items():
            difficulty=key.split('/')[0]
            groups[('input',difficulty)].append(item['metrics'])
        for key,item in data.get('relaxed',{}).items():
            groups[('UFF',key.split('/')[0])].append(item['metrics'])
        for checkpoint,items in data.get('generated',{}).items():
            for key,item in items.items():
                difficulty=key.split('/')[0]
                metrics={**item['metrics'],'input_metrics':data['inputs'][key]['metrics']}
                metrics['native_physical_category']=physical_category(data.get('native'))
                groups[(checkpoint,difficulty)].append(metrics)
    stats=[]
    for (method,difficulty),rows in sorted(groups.items()):
        rmsds=[r['free_rmsd'] for r in rows if 'free_rmsd' in r and r.get('graph_preserved')]
        entry={'method':method,'difficulty':difficulty,'evaluated':len(rows),
               'expected':plan['actual_cases']*plan['seeds'],
               'pocket':dict(Counter(physical_category(r) for r in rows)),
               'full_protein':dict(Counter(physical_category(r,'full_protein') for r in rows)),
               'graph_preserved':sum(r.get('graph_preserved',False) for r in rows),
               'median_free_rmsd':float(np.median(rmsds)) if rmsds else None}
        if method in CHECKPOINTS:
            entry['physical_transitions']=dict(Counter(f"{physical_category(r['input_metrics'])}->{physical_category(r)}" for r in rows))
            entry['free_rmsd_improved']=sum(r.get('graph_preserved',False) and
                     r.get('free_rmsd',float('inf')) < r['input_metrics'].get('free_rmsd',0) for r in rows)
            entry['primary_success']=sum(r.get('graph_preserved',False) and r.get('held_max_drift',1) <= .001
                     and physical_category(r)=='valid' and r.get('free_rmsd',float('inf')) <
                     r['input_metrics'].get('free_rmsd',0) for r in rows)
            source_valid=[r for r in rows if r['native_physical_category']=='valid']
            entry['native_valid_subset']={'evaluated':len(source_valid),
                     'pocket_valid':sum(physical_category(r)=='valid' for r in source_valid),
                     'free_rmsd_improved':sum(r.get('graph_preserved',False) and
                         r.get('free_rmsd',float('inf')) < r['input_metrics'].get('free_rmsd',0) for r in source_valid)}
        stats.append(entry)
    generation={cp:sum((root/'draws'/cp/c['case_id']/i['difficulty']/str(i['seed'])/'result.json').exists()
                   for c in plan['cases'] for i in c['inputs']) for cp in CHECKPOINTS}
    status=('complete_with_missing_evaluation' if incomplete_cases else 'complete') if complete_cases==plan['actual_cases'] else 'running'
    summary={'status':status,'evaluation_incomplete_cases':incomplete_cases,
             'cases':plan['actual_cases'],'evaluation_complete_cases':complete_cases,
             'generation_outputs':generation,'planned_outputs':plan['planned_outputs'],
             'native_evaluated':len(natives),
             'native_pocket':dict(Counter(physical_category(r) for r in natives)),
             'native_full_protein':dict(Counter(physical_category(r,'full_protein') for r in natives)),
             'groups':stats}
    save(root/'summary.json',summary)
    lines=[f"# GPU {plan['physical_gpu']} 조건부 pose 복구 실험",'',f"상태: {summary['status']}. 복합체 {plan['actual_cases']}개, 물리 평가 완료 {complete_cases}개.",'',
           '전체 graph를 고정하고 선택한 자유 영역의 좌표만 재생성했다. 두 모델은 동일한 흐트러뜨린 입력을 받았다. LLM 호출과 추가 학습은 없다.','',
           f"원본 물리 검사: pocket {summary['native_pocket']}, 전체 protein {summary['native_full_protein']} (측정 {len(natives)}개).",'',
           '| 방법 | 오차 수준 | 평가/예정 | Pocket 통과 | 전체 protein 통과 | 자유 영역 RMSD 중앙값 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for g in stats:
        value=f"{g['median_free_rmsd']:.3f} Å" if g['median_free_rmsd'] is not None else '미측정'
        lines.append(f"| {g['method']} | {g['difficulty']} | {g['evaluated']}/{g['expected']} | {g['pocket'].get('valid',0)} | {g['full_protein'].get('valid',0)} | {value} |")
    lines+=['','누락과 build 실패를 통과로 처리하지 않는다. Physical validity와 graph/좌표 보존, RMSD 복구는 서로 다른 지표다. 전체 protein 추가 검사는 물과 보조인자의 고정 좌표도 포함한다.','',
            '원소·전하·결합 등 discrete features는 샘플링 중 고정한다. 입체화학은 별도 tensor 조건이 없으므로 생성 XYZ에서 계산해 출력 계약으로 검사한다. 잘못된 입체화학을 parent annotation으로 덮어쓰지 않는다.','',
            'RMSD는 고정된 protein frame에서 보호 원자의 대응을 유지하는 stereograph symmetry를 고려한다(maxMatches=4096). Ligand를 정답에 정렬하지 않는다. Atom map 기준 RMSD도 별도 저장한다.','',
            'UFF 대조군은 ligand-only geometry 최적화다. 보호 원자의 좌표는 고정하지만 protein energy term을 사용하지 않아 pocket-aware 최적화와 동일하지 않다.','',
            '원본부터 물리 검사에 실패한 사례도 유지하고 원본/입력/출력의 검사별 결과를 evaluations에 저장한다. 평가는 후보를 정답 RMSD로 선택하지 않고 전체 예정 draw를 집계한다.','',
            f"선정: {plan['selection']}. 서로 다른 복합체가 같은 protein/scaffold를 가질 수 있는 설정은 명시적으로 기록한다. FM pretraining overlap 및 sequence cluster 일반화는 미확인이다.",'',
            f"원본 stereo를 XYZ에서 보완한 설정: {plan.get('coordinate_stereo_baseline',False)}. 명시된 원본 stereo와 충돌하는 사례는 제외한다. 생성물은 원본 XYZ로 정의한 stereo까지 보존해야 한다.",'',
            f"최대 배치는 {plan['batch_cap']}이다. 큰 pocket이나 OOM에서는 낮추며, 한 배치는 같은 복합체의 서로 다른 입력으로 구성한다. 혼합 크기 배치는 padding 영향 검증이 끝나지 않아 사용하지 않는다. 배치·실패·최대 메모리는 phases/*/batches.jsonl에 저장한다.",'',
            f"GPU {plan['physical_gpu']}에서 두 checkpoint를 병행하므로 latency를 단독 실행 성능으로 해석하지 않는다.",'']
    (root/'REPORT.md').write_text('\n'.join(lines))


def run(args):
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    plan=prepare(args)
    plan.update(physical_gpu=args.gpu,runtime_script_sha256=sha256_file(__file__),
                backend_sha256=sha256_file(ROOT/'src/dirmo/generation/pose_repair.py'),
                stereo_contract='Derived from generated XYZ and checked after sampling; no separate stereo input feature')
    save(root/'study.json',plan)
    device=resolve_device(args.gpu,inventory())
    save(root/'gpu.json',device)
    env=worker_environment(device,str(ROOT/'src'))
    workers={}
    streams=[]
    for checkpoint in CHECKPOINTS:
        log=root/'phases'/checkpoint/'generation.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        stream=log.open('a');streams.append(stream)
        command=[sys.executable,__file__,'--checkpoint',checkpoint,'--gpu',args.gpu,
                 '--output',str(root),'--checkpoints',str(args.checkpoints)]
        workers[checkpoint]=subprocess.Popen(command,env=env,stdout=stream,stderr=stream)
    cpu_env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    submitted={}
    def evaluation_task(case_id):
        log=root/'evaluations'/f'{case_id}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,__file__,'--evaluate-case','--case-id',case_id,'--output',str(root)]
        case=read(root/'cases'/case_id/'case.json')
        timeout=1800 if Path(case['protein_path']).stat().st_size>2_000_000 else 900
        # Retry an incomplete evaluator once, preserving already saved stages.
        for attempt in range(2):
            try:
                with log.open('a') as stream:
                    result=subprocess.run(command,env=cpu_env,stdout=stream,stderr=stream,timeout=timeout)
                if result.returncode==0:
                    return {'case_id':case_id,'status':'complete'}
            except subprocess.TimeoutExpired:
                pass
        data=read(root/'evaluations'/f'{case_id}.json',{'case_id':case_id})
        data.update(complete=True,evaluation_incomplete=True,reason='Evaluator failed/timed out twice; missing stages are explicit')
        save(root/'evaluations'/f'{case_id}.json',data)
        return {'case_id':case_id,'status':'complete_with_missing_metrics'}
    save(root/'controller.json',{'status':'running','pid':os.getpid(),
         'gpu_worker_pids':{k:p.pid for k,p in workers.items()},'physical_gpu':args.gpu})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            while True:
                for case in plan['cases']:
                    key=case['case_id']
                    if key in submitted:
                        continue
                    if read(root/'evaluations'/f'{key}.json',{}).get('complete'):
                        submitted[key]=None
                    elif all((root/'phases'/cp/f'{key}.done.json').exists() for cp in CHECKPOINTS) and (
                        not plan.get('baseline_output') or
                        read(Path(plan['baseline_output'])/'evaluations'/f'{key}.json',{}).get('complete')):
                        submitted[key]=pool.submit(evaluation_task,key)
                aggregate(root)
                done=sum(f is None or f.done() for f in submitted.values())
                emit(status='progress',gpu_workers={k:p.poll() for k,p in workers.items()},
                     cpu_cases_submitted=len(submitted),cpu_cases_done=done,total_cases=len(plan['cases']))
                if all(p.poll() is not None for p in workers.values()):
                    failed={k:p.returncode for k,p in workers.items() if p.returncode}
                    if failed:
                        save(root/'controller.json',{'status':'gpu_worker_failed','failed':failed,'pid':os.getpid()})
                        raise RuntimeError(f'GPU workers failed: {failed}; resume is supported')
                    if done==len(plan['cases']):
                        break
                time.sleep(20)
            for future in submitted.values():
                if future:
                    future.result()
            aggregate(root)
            save(root/'controller.json',{'status':read(root/'summary.json')['status'],'pid':os.getpid(),'physical_gpu':args.gpu})
            emit(status='complete',report=str(root/'REPORT.md'))
        finally:
            for stream in streams:
                stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='4')
    parser.add_argument('--cases',type=int,default=50)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dataset',type=Path,default=Path('/LocalData1/uslee/datasets/PDBbind2019/refined-set'))
    parser.add_argument('--extra-dataset',type=Path,action='append',default=[])
    parser.add_argument('--allow-related-complexes',action='store_true')
    parser.add_argument('--coordinate-stereo-baseline',action='store_true')
    parser.add_argument('--checkpoints',type=Path,default=Path('/LocalData1/uslee/checkpoints/FLOWR_root'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',choices=list(CHECKPOINTS))
    parser.add_argument('--evaluate-case',action='store_true')
    parser.add_argument('--case-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if not 1<=args.cases<=5000 or not 1<=args.seeds<=20 or args.batch_size not in (1,2,4,8,16,32):
        parser.error('Cases/seeds/batch outside supported experiment limits')
    if not 1<=args.workers<=8 or not 1<=args.steps<=256:
        parser.error('Workers/steps outside supported experiment limits')
    if args.checkpoint:
        generate(args)
    elif args.evaluate_case:
        evaluate_case(args)
    elif args.report_only:
        aggregate(args.output)
    elif args.prepare_only:
        prepare(args)
    else:
        import fcntl
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args)


if __name__=='__main__':
    main()
