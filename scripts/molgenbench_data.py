"""Leakage audit and assay-matched pair feasibility for MolGenBench v4.

The public series identifier is NOT silently treated as an assay identifier.
Where available, exact molecule/target/endpoint/value matches in local ChEMBL37
provide separately attributed assay evidence; this is not a reconstruction of
the original ChEMBL33 extraction. Evaluation actives never enter proposal rules.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from itertools import combinations
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold
from router_contract_pilot import ROOT, read, save, now, log, digest, sha256_file
from dirmo.chemistry.graphs import read_sdf

DATA = Path('/LocalData1/uslee/datasets/molgenbench-v4')
CHEMBL = Path('/LocalData1/uslee/datasets/chembl_37/chembl_37_sqlite/chembl_37.db')

def identity(mol):
    mol = Chem.RemoveHs(Chem.Mol(mol))
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def fragments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not 6 <= mol.GetNumHeavyAtoms() <= 64: return []
    result = set()
    for _, chain in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = chain.split('.')
        if len(parts) != 2: continue
        for rest, small in (parts, parts[::-1]):
            n = Chem.MolFromSmiles(small).GetNumHeavyAtoms()
            if 1 <= n <= min(8, mol.GetNumHeavyAtoms() // 3):
                result.add((rest.replace('[*:1]', '*'), small.replace('[*:1]', '*')))
    return sorted(result)

def crossdock(root):
    """Exact graph and source-PDB audit against the actual local train index."""
    dest = root / 'crossdock_train_index.json'
    index = Path('/LocalData1/uslee/datasets/processed_crossdock_noH_full_temp/train.npz')
    # The large NPZ hash is computed once; the derived cache is local to this study.
    with np.load(index) as z: names = z['names'].tolist()
    names_hash = digest(names)
    if dest.exists():
        result = read(dest)
        if result['names_sha256'] != names_hash: raise ValueError('CrossDocked index changed')
        return result
    pairs = [n.split('.pdb_', 1) for n in names]
    pdbs = set()
    for protein, ligand in pairs:
        pdbs.add(Path(protein).name.split('_')[0].lower())
        fields = Path(ligand).name.split('_')
        if len(fields) > 3: pdbs.add(fields[3].lower())
    raw = Path('/LocalData1/uslee/datasets/crossdocked_pocket10')
    def one(ligand):
        try: return identity(read_sdf(raw / ligand)), None
        except Exception as exc: return None, dict(path=ligand, reason=str(exc))
    graphs, failed = set(), []
    ligands = sorted({p[1] for p in pairs})
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (graph, error) in enumerate(pool.map(one, ligands)):
            if graph: graphs.add(graph)
            if error: failed.append(error)
            if i % 10000 == 0: log('crossdock_graph_audit', completed=i, total=len(ligands))
    result = dict(time_kst=now(), index=str(index), names_sha256=names_hash, rows=len(names),
                  source_pdbs=sorted(pdbs), exact_graphs=sorted(graphs), failed=failed)
    save(dest, result)
    return result

def inventory(data):
    manifest = read(data / 'sources.json')
    if read(data / 'download_status.json')['status'] != 'complete': raise ValueError('Download incomplete')
    # Verify every selected downloaded member before processing.
    for entry in manifest['files']:
        if sha256_file(data / entry['path']) != entry['sha256']: raise ValueError(f"Changed source {entry['path']}")
    code = data / 'official-code'
    overlap = read(code / 'sup_info/UniprotIDs_duplicated_with_crossdock2020.json')
    pdbs = {r['uniprot']:r['pdb_id'] for r in csv.DictReader((code/'sup_info/uniprot_map_pdbid.csv').open())}
    records = []
    for table in sorted((data/'MolGenBench_Version4').glob('*/reference_active_molecules/Hit2Lead/top5_common_scaffold_info.csv')):
        target = table.parents[2].name
        for row in csv.DictReader(table.open()):
            series = row['SeriseID']
            pose = table.parent/f'{target}_{series}_reference_ligand_pose_with_h.sdf'
            # FilteredMols is deliberately NOT copied to the generator manifest.
            records.append(dict(target=target, series=series, scaffold_smarts=row['Scaffold'],
                reference_sdf=str(pose), protein_pdb=str(table.parents[2]/f'{target}_prep.pdb'),
                pdb_id=pdbs.get(target), official_crossdock_target_overlap=target in overlap,
                source_table_sha256=sha256_file(table)))
    if len({r['target'] for r in records}) != 120 or len(records) != 600:
        raise ValueError(f'Unexpected v4 inventory: {len(records)} series')
    return records

def audit(args):
    root=args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    frozen=root/'target_split.json'
    if frozen.exists(): return read(frozen)
    records=inventory(args.data); cd=crossdock(root)
    graphs=set(cd['exact_graphs']); pdbs=set(cd['source_pdbs']); checked=[]
    for record in records:
        r=dict(record); reasons=[]
        if r['official_crossdock_target_overlap']: reasons.append('official_crossdock_target_overlap')
        if r['pdb_id'] in pdbs: reasons.append('local_crossdock_source_pdb_overlap')
        try:
            mol=read_sdf(r['reference_sdf']); smi=identity(mol)
            r.update(parent_smiles=smi, parent_heavy_atoms=mol.GetNumHeavyAtoms())
            if smi in graphs: reasons.append('local_crossdock_exact_parent_graph_overlap')
            if not 12<=mol.GetNumHeavyAtoms()<=64: reasons.append('outside_fm_atom_count')
            query=Chem.MolFromSmarts(r['scaffold_smarts'])
            matches=mol.GetSubstructMatches(query, uniquify=True) if query is not None else ()
            if not matches: reasons.append('scaffold_not_matched')
            r['scaffold_matches']=len(matches)
            if matches and len(matches[0])==mol.GetNumHeavyAtoms(): reasons.append('scaffold_covers_entire_parent')
        except Exception as exc: reasons.append(f'input_error:{exc}')
        r['exclusions']=reasons;checked.append(r)
    # Select target and one series without consulting PB, Vina or affinity.
    targets=sorted({r['target'] for r in checked if not r['exclusions']},key=lambda t:digest([args.seed,t]))
    selected=[]
    for target in targets[:args.targets]:
        candidates=[r for r in checked if r['target']==target and not r['exclusions']]
        selected.append(min(candidates,key=lambda r:digest([args.seed,r['target'],r['series']])))
    heldout={r['target'] for r in selected}; alltargets={r['target'] for r in records}
    result=dict(time_kst=now(),seed=args.seed,requested_targets=args.targets,selected=selected,
        evaluation_targets=sorted(heldout),development_targets=sorted(alltargets-heldout),
        eligible_targets=len(targets),source_manifest_sha256=sha256_file(args.data/'sources.json'),
        selection='Hash order of target then series, outcome-blind; no replacement after menu or metric failures',
        crossdock=dict(train_rows=cd['rows'],unique_graphs=len(graphs),unreadable_train_files=len(cd['failed']),
                      exact_graph_audit_complete=not cd['failed']),
        fm_pretraining_overlap='unknown; checkpoint training membership unavailable',
        frozen_router_historical_mmp_exposure='unknown; old aggregate MMP table has no pair/assay provenance',
        existing_mmp=dict(path=str(args.rules),sha256=sha256_file(args.rules),
            use_for_new_proposals=False,reason='Aggregate table cannot exclude evaluation-target evidence'),
        rule_policy='New structure-only MMP rules with development-target witnesses; no activity direction/count filters; heldout molecule identities excluded across targets',
        regime='full-reference-ligand SBMO adaptation; not official scaffold-only H2L')
    save(root/'inventory.json',checked);save(frozen,result)
    log('target_split_frozen',selected=len(selected),eligible=len(targets))
    return result

def active_records(data, target):
    path=data/'MolGenBench_Version4'/target/'reference_active_molecules'/f'{target}_reference_active_molecules.sdf'
    for index,mol in enumerate(Chem.SDMolSupplier(str(path))):
        if mol is None:
            yield dict(target=target,index=index,error='invalid_sdf');continue
        name=mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        props=mol.GetPropsAsDict()
        try:
            smi=identity(mol);parts=name.split('_');series=parts[1] if len(parts)>1 else None
            value=float(props.get('Affinity',float('nan')))
            yield dict(target=target,series=series,index=index,name=name,smiles=smi,
                inchi_key=Chem.MolToInchiKey(Chem.MolFromSmiles(smi)),
                affinity=value if math.isfinite(value) and value>0 else None,
                endpoint=props.get('Affinity Type'),unit=props.get('Affinity Unit'),
                explicit_assay_id=next((str(props[k]) for k in ('assay_id','Assay ID','CHEMBL_ASSAY_ID') if k in props),None))
        except Exception as exc: yield dict(target=target,index=index,error=str(exc))

def evidence(args):
    root=args.output.resolve(); split=read(root/'target_split.json')
    if (root/'evidence_complete.json').exists(): return
    held=set(split['evaluation_targets']); targets=sorted(held|set(split['development_targets']))
    top={(r['target'],r['series']) for r in read(root/'inventory.json')}
    records=[];counts=Counter();evaluation_graphs=set()
    for target in targets:
        for row in active_records(args.data,target):
            counts['all_records']+=1
            if 'error' in row:counts['invalid']+=1;continue
            if target in held:evaluation_graphs.add(row['smiles'])
            counts['explicit_assay_id']+=row['explicit_assay_id'] is not None
            if (target,row['series']) in top:records.append(row)
    save(root/'activity_inventory.json',dict(time_kst=now(),counts=dict(counts),top_series_records=len(records),
        note='Series IDs alone are not certified assay identifiers; exact local ChEMBL37 witnesses are separately attributed.'))
    save(root/'evaluation_only_actives.json',[r for r in records if r['target'] in held])
    development=[r for r in records if r['target'] not in held and r['smiles'] not in evaluation_graphs]
    save(root/'development_records.json',development)
    # Structure-only transformation vocabulary; never use heldout affinities or
    # signs/consistency from the original aggregated activity-derived rules.
    core_groups=defaultdict(dict)
    for row in development:
        for core,frag in fragments(row['smiles']):
            core_groups[(row['target'],row['series'],core)].setdefault(frag,row['name'])
    rules={}
    for (target,series,core),members in sorted(core_groups.items()):
        for left,right in combinations(sorted(members),2):
            key=(left,right)
            item=rules.setdefault(key,dict(lhs=left,rhs=right,witness_count=0,witnesses=[]))
            item['witness_count']+=1
            if len(item['witnesses'])<3:
                item['witnesses'].append(dict(target=target,series=series,
                    molecules=[members[left],members[right]],retained_fragment=core))
    save(root/'development_rules.json',list(rules.values()))
    log('development_rules',rules=len(rules),records=len(development))
    # Supplementary real assay evidence from the read-only local ChEMBL37 DB.
    conn=sqlite3.connect(f'file:{args.chembl}?mode=ro',uri=True)
    conn.execute('PRAGMA cache_size=-100000')
    target_ids={t:{x[0] for x in conn.execute('''SELECT DISTINCT tc.tid FROM target_components tc
        JOIN component_sequences cs ON cs.component_id=tc.component_id
        JOIN target_dictionary td ON td.tid=tc.tid
        WHERE cs.accession=? AND td.target_type='SINGLE PROTEIN' ''',(t,))} for t in split['development_targets']}
    cache={};groups=defaultdict(dict); evidence_counts=Counter()
    for i,row in enumerate(development):
        if row['unit']!='nM' or row['affinity'] is None or row['endpoint'] not in ('Ki','Kd','IC50','EC50'):
            evidence_counts['unsupported_endpoint_or_units']+=1;continue
        key=(row['inchi_key'],row['target'])
        if key not in cache:
            cache[key]=conn.execute('''SELECT ac.activity_id,ac.assay_id,a.tid,ac.standard_type,
                ac.standard_value,ac.standard_units,ac.standard_relation,a.chembl_id
                FROM compound_structures cs JOIN activities ac ON ac.molregno=cs.molregno
                JOIN assays a ON a.assay_id=ac.assay_id WHERE cs.standard_inchi_key=?
                AND ac.standard_relation='=' AND ac.standard_units='nM'
                AND ac.standard_value>0 AND ac.data_validity_comment IS NULL
                AND a.confidence_score=9 AND a.variant_id IS NULL''',(row['inchi_key'],)).fetchall()
        matches=[v for v in cache[key] if v[2] in target_ids[row['target']] and v[3]==row['endpoint']
                 and math.isclose(v[4],row['affinity'],rel_tol=1e-6,abs_tol=1e-8)]
        if not matches: evidence_counts['no_exact_chembl37_witness']+=1;continue
        evidence_counts['records_with_exact_witness']+=1
        for activity,assay,tid,endpoint,value,unit,relation,chembl_assay in matches:
            group=(row['target'],chembl_assay,endpoint)
            groups[group].setdefault(row['smiles'],dict(**row,activity_id=activity,chembl_assay=chembl_assay,
                p_activity=9.-math.log10(value),measurement_source='ChEMBL37 exact identity/target/type/value witness'))
        if i%1000==0: log('assay_evidence_progress',completed=i,total=len(development))
    pairs=[]
    for (target,assay,endpoint),members in sorted(groups.items()):
        core_map=defaultdict(list)
        for smi,row in members.items():
            for core,frag in fragments(smi): core_map[core].append((frag,row))
        seen=set()
        for core,items in sorted(core_map.items()):
            for (fa,a),(fb,b) in combinations(items,2):
                keys=tuple(sorted([a['smiles'],b['smiles']]))
                if fa==fb or keys in seen or abs(a['p_activity']-b['p_activity'])<1e-8:continue
                seen.add(keys)
                if a['p_activity']>b['p_activity']:a,b=b,a;fa,fb=fb,fa
                pairs.append(dict(target=target,assay_id=assay,endpoint=endpoint,unit='p'+endpoint,
                    initial_smiles=a['smiles'],optimized_smiles=b['smiles'],
                    initial_p_activity=a['p_activity'],optimized_p_activity=b['p_activity'],
                    delta_p_activity=b['p_activity']-a['p_activity'],transformation=fa+'>>'+fb,
                    retained_fragment=core,activity_ids=[a['activity_id'],b['activity_id']],
                    molgenbench_record_names=[a['name'],b['name']],split='development',
                    provenance='MolGenBench molecules with separate exact-matching ChEMBL37 same-assay measurements',
                    fm_mode_label=None))
    with (root/'activity_pairs.jsonl').open('w') as stream:
        import json
        for p in pairs:stream.write(json.dumps(p,ensure_ascii=False)+'\n')
    save(root/'pair_examples.json',pairs[:5])
    save(root/'evidence_complete.json',dict(time_kst=now(),activity_counts=dict(counts),
        development_records=len(development),excluded_shared_eval_graphs=sum(r['target'] not in held and r['smiles'] in evaluation_graphs for r in records),
        rules=len(rules),assay_evidence=dict(evidence_counts),same_assay_mmp_pairs=len(pairs),
        pair_targets=len({p['target'] for p in pairs}),pair_endpoints=dict(Counter(p['endpoint'] for p in pairs)),
        crossdock_overlap_in_development_allowed=True,
        training_feasibility='Chemical edit ranking supervision only; no FM mode or pose labels. Split development targets further before any future tuning.',
        chembl_database=str(args.chembl),chembl_version=37,original_chembl33_assay_reconstruction=False))
    original_rules=Counter()
    for row in csv.DictReader(args.rules.open()):
        original_rules['rows']+=1
        left,right=row['transform'].split('>>')
        original_rules['also_witnessed_in_development_structures']+=tuple(sorted([left,right])) in rules
    save(root/'mmp_provenance_audit.json',dict(time_kst=now(),original_table=str(args.rules),
        original_table_sha256=sha256_file(args.rules),counts=dict(original_rules),
        original_assay_and_target_provenance='unavailable; not certified free of evaluation evidence',
        used_for_current_proposals=False,development_rules_sha256=sha256_file(root/'development_rules.json'),
        evaluation_target_witnesses_in_new_rules=0,shared_evaluation_graphs_excluded=True))
    lines=['# MolGenBench 데이터·학습 쌍 점검',f'작성: {now()}','',
        f"공식 v4: 120 타깃 / 600 series. 외부 평가 타깃 {len(split['evaluation_targets'])}개를 결과 확인 전에 고정했다.",
        f"CrossDocked 로컬 train {split['crossdock']['train_rows']}행의 정확한 분자 그래프와 source PDB를 대조했다. 읽기 실패 {split['crossdock']['unreadable_train_files']}개.",
        'FM 사전학습 전체 중복, 유사 단백질·동족체 중복, 동결 Router의 과거 MMP 증거 중복은 확인되지 않았다.',
        f"활성 SDF {counts['all_records']}행 중 명시적 assay ID가 있는 행은 {counts['explicit_assay_id']}개다. series ID를 assay ID로 간주하지 않았다.",
        f"개발용 구조 {len(development)}행에서 구조 기반 MMP 규칙 {len(rules)}개를 만들었다. 평가 타깃의 활성값/구조는 생성기 규칙에 사용하지 않았다.",
        f"같은 타깃·assay·endpoint의 ChEMBL37 측정으로 확인한 MMP 쌍 {len(pairs)}개, 타깃 {len({p['target'] for p in pairs})}개.",
        'ChEMBL37 근거는 분자 InChIKey, 단일 단백질 타깃, endpoint, nM 값의 정확한 대응 및 relation =, confidence 9, 비변이 assay를 요구했다. MolGenBench 원본 ChEMBL33 assay 복원을 의미하지 않는다.',
        'Ki/Kd/IC50/EC50은 각각 분리했다. 같은 assay의 서로 다른 분자 비교이며, 화학 변형의 인과 효과나 생성 모드 정답이라고 주장하지 않는다.',
        '새 분자의 실험 활성은 알 수 없다. 향후 학습 전에 개발용 타깃을 train/validation으로 추가 분리하고 scaffold/유사 단백질 중복을 확인해야 한다.',
        '', '파일: target_split.json, inventory.json, crossdock_train_index.json, mmp_provenance_audit.json, activity_inventory.json, evidence_complete.json, activity_pairs.jsonl, pair_examples.json.']
    (root/'DATA_AUDIT.md').write_text('\n\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['audit','evidence','all'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data',type=Path,default=DATA)
    ap.add_argument('--targets',type=int,default=20);ap.add_argument('--seed',type=int,default=20261105)
    ap.add_argument('--rules',type=Path,default=Path('/home/uslee/code/dataset_analyze/mmp_local_rules.csv'))
    ap.add_argument('--chembl',type=Path,default=CHEMBL)
    args=ap.parse_args();RDLogger.DisableLog('rdApp.*')
    if args.command in ('audit','all'):audit(args)
    if args.command in ('evidence','all'):evidence(args)

if __name__=='__main__':main()
