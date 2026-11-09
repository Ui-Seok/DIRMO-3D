"""CPU-only E5 audit: fixed heavy coordinates, explicit/implicit H, atom attribution.

Writes a new run only. Does not import torch or change original experiment files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
from zoneinfo import ZoneInfo

for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import rdkit
import prolif as plf
from prolif.io import MoleculeStandardizer

RDLogger.DisableLog('rdApp.*')
E = Path('/LocalData1/uslee/experiments/DIRMO-3D/clamp_gpu1_20261021_v2/full')
R = Path('/LocalData1/uslee/experiments/DIRMO-3D/refine_20261022_v1')
OLD = Path('/LocalData1/uslee/experiments/DIRMO-3D/ifp_20261022_v1')
SERIES = ['prior', 'fm_core', 'fm_whole', 'uff_lig', 'uff_pkt', 'uff_pkt_soft', 'uff_pkt_tether']
METHODS = ['legacy', 'explicit_raw', 'explicit_prepared', 'implicit_prepared']
DEFAULT_TYPES = ['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic',
                 'Cationic', 'CationPi', 'PiCation', 'VdWContact']
TYPES = [*DEFAULT_TYPES, 'XBAcceptor', 'XBDonor']
GROUPS = {
    'all': set(TYPES), 'non_vdw': set(TYPES)-{'VdWContact'},
    'common_non_vdw': set(DEFAULT_TYPES)-{'VdWContact'},
    'polar_pi': set(TYPES)-{'VdWContact', 'Hydrophobic'},
    'hydrogen_bond': {'HBDonor','HBAcceptor'}, 'HBDonor': {'HBDonor'},
    'HBAcceptor': {'HBAcceptor'}, 'ionic': {'Anionic','Cationic'},
    'pi': {'PiStacking','CationPi','PiCation'},
    'halogen': {'XBAcceptor','XBDonor'}, 'hydrophobic': {'Hydrophobic'},
}


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def emit(**data):
    print(json.dumps({'time_kst': now(), **data}, ensure_ascii=False), flush=True)


def read_mol(path):
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
    if mol is None:
        raise ValueError('RDKit SDF parse failed')
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError('Expected original heavy-only ligand')
    if mol.GetNumConformers() != 1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError('Missing/nonfinite coordinates')
    return mol


def normalize(mol, native):
    expected = [a.GetAtomMapNum() for a in native.GetAtoms()]
    actual = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    if all(a.GetAtomMapNum()==0 for a in mol.GetAtoms()):
        # FLOWR output SDFs omit atom maps. Their ordered typed graph must match
        # the conditioning graph exactly before reusing that graph's indices.
        out = Chem.Mol(mol)
        mapping_method = 'verified_ordered_typed_graph'
    elif len(actual)==mol.GetNumAtoms() and set(actual)==set(expected) and 0 not in actual:
        out = Chem.RenumberAtoms(mol, [actual[n] for n in expected])
        mapping_method = 'explicit_atom_maps'
    else:
        raise ValueError('Partial/nonunique atom maps without ordered-graph evidence')
    def topology(m):
        return ([tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic(),
                        a.GetTotalNumHs(includeNeighbors=True))) for a in m.GetAtoms()],
                sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()), str(b.GetBondType()))
                       for b in m.GetBonds()))
    if topology(out) != topology(native):
        raise ValueError('Connectivity/element/charge/H-count differs from native state')
    for atom, number in zip(out.GetAtoms(), expected):
        atom.SetAtomMapNum(number)
    out.SetProp('_IFP_atom_mapping_method',mapping_method)
    return out


def add_hydrogens(mol):
    n = mol.GetNumAtoms()
    xyz = mol.GetConformer().GetPositions().copy()
    prepared = Chem.AddHs(mol, addCoords=True)
    if not np.isfinite(prepared.GetConformer().GetPositions()).all():
        raise ValueError('Nonfinite H coordinates')
    ff = AllChem.UFFGetMoleculeForceField(prepared)
    if ff is None:
        raise ValueError('H-only UFF unavailable')
    for i in range(n):
        ff.AddFixedPoint(i)
    code = int(ff.Minimize(maxIts=200))
    delta = float(np.linalg.norm(prepared.GetConformer().GetPositions()[:n]-xyz, axis=1).max())
    if delta > 1e-10:
        raise ValueError(f'Heavy-coordinate movement during H preparation: {delta}')
    if not np.isfinite(prepared.GetConformer().GetPositions()).all():
        raise ValueError('Nonfinite coordinates after H-only relaxation')
    return prepared, {'added_H': prepared.GetNumAtoms()-n, 'H_only_UFF_code': code,
                      'heavy_max_drift_A': delta, 'formal_charge': Chem.GetFormalCharge(prepared)}


def heavy_parent(mol, indices):
    heavy = set()
    for index in indices:
        atom = mol.GetAtomWithIdx(int(index))
        if atom.GetAtomicNum() != 1:
            heavy.add(atom.GetIdx())
        else:
            heavy.update(a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum()!=1)
    return sorted(heavy)


def events_of(mol, protein, fp):
    """Keep parent heavy-atom identities, geometry, and raw residue keys for parity."""
    ligand = plf.Molecule.from_rdkit(mol)
    data = fp.generate(ligand, protein, metadata=True)
    events, raw_pairs = {}, set()
    for (_, resid), inter in data.items():
        pres = protein[resid]
        pglobal = {int(a.GetUnsignedProp('mapindex')): a.GetIdx() for a in pres.GetAtoms()}
        for typ, occurrences in inter.items():
            typ = typ.removeprefix('Implicit')
            raw_pairs.add(f'{resid}|{typ}')
            for rec in occurrences:
                li = tuple(int(x) for x in rec['parent_indices']['ligand'])
                pi = tuple(int(x) for x in rec['parent_indices']['protein'])
                pi_local = [pglobal[i] for i in pi]
                # Heavy-only VdW contacts give the same definition with/without H.
                if typ == 'VdWContact' and (any(mol.GetAtomWithIdx(i).GetAtomicNum()==1 for i in li)
                        or any(pres.GetAtomWithIdx(i).GetAtomicNum()==1 for i in pi_local)):
                    continue
                lh = heavy_parent(mol, li)
                ph = heavy_parent(pres, pi_local)
                lm = tuple(sorted(mol.GetAtomWithIdx(i).GetAtomMapNum() for i in lh))
                pn = tuple(sorted(pres.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip() for i in ph))
                if not lm or not pn:
                    raise ValueError('Interaction cannot be attributed to heavy atoms')
                key = (str(resid), typ, lm, pn)
                geo = {k: float(v) for k,v in rec.items()
                       if k not in ('indices','parent_indices') and isinstance(v,(int,float,np.number))
                       and np.isfinite(v)}
                event = {'residue': str(resid), 'type': typ, 'ligand_maps': list(lm),
                         'protein_atom_names': list(pn), 'geometry': geo}
                if key not in events or geo.get('distance',float('inf')) < events[key]['geometry'].get('distance',float('inf')):
                    events[key] = event
    return sorted(events.values(), key=lambda e:(e['residue'],e['type'],e['ligand_maps'],e['protein_atom_names'])), sorted(raw_pairs)


def prepare_protein(path):
    raw = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    if raw is None:
        raise ValueError('Protein PDB parse failed')
    if not any(a.GetAtomicNum()==1 for a in raw.GetAtoms()):
        raise ValueError('Protein has no provided explicit H')
    legacy = plf.Molecule(Chem.Mol(raw))
    legacy_covered = {int(a.GetUnsignedProp('mapindex')) for r in legacy.residues.values() for a in r.GetAtoms()}
    omitted = sorted(set(range(raw.GetNumAtoms()))-legacy_covered)
    template_input = Chem.Mol(raw)
    if omitted:
        # ProLIF's connectivity-based split can overwrite disconnected fragments
        # carrying the same PDB residue id. Group by PDB identity instead, keeping
        # ALL atoms. The legacy control deliberately retains its original path.
        from prolif.residue import Residue
        for a in template_input.GetAtoms():
            a.SetUnsignedProp('mapindex',a.GetIdx())
        groups = defaultdict(list)
        for a in template_input.GetAtoms():
            ri = a.GetPDBResidueInfo()
            groups[(ri.GetChainId(),ri.GetResidueNumber(),ri.GetInsertionCode(),ri.GetResidueName())].append(a.GetIdx())
        residues = []
        for ids in groups.values():
            keep = set(ids)
            em = Chem.RWMol(template_input)
            for i in range(template_input.GetNumAtoms()-1,-1,-1):
                if i not in keep:
                    em.RemoveAtom(i)
            res = em.GetMol()
            res.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(res)
            residues.append(Residue(res))
        template_input = plf.Molecule(template_input,residues=residues)
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        prepared = MoleculeStandardizer()(template_input)
    max_drift = 0.
    heavy_keys = set()
    all_keys = set()
    state = []
    for res in prepared.residues.values():
        xyz = res.GetConformer().GetPositions()
        atoms = []
        for a in res.GetAtoms():
            parent = int(a.GetUnsignedProp('mapindex'))
            all_keys.add(parent)
            max_drift = max(max_drift, float(np.linalg.norm(xyz[a.GetIdx()]-raw.GetConformer().GetPositions()[parent])))
            if a.GetAtomicNum()!=1:
                heavy_keys.add(parent)
            atoms.append({'parent_index':parent,'name':a.GetPDBResidueInfo().GetName().strip(),
                          'element':a.GetSymbol(),'charge':a.GetFormalCharge(),
                          'explicit_H_neighbors':sum(n.GetAtomicNum()==1 for n in a.GetNeighbors())})
        state.append({'residue':str(res.resid), 'atoms':atoms})
    if all_keys != set(range(raw.GetNumAtoms())) or max_drift > 1e-10:
        raise ValueError('Protein atom/coordinate correspondence changed')
    return legacy, prepared, {
        'path':str(path),'sha256':digest(path),'provided_H':sum(a.GetAtomicNum()==1 for a in raw.GetAtoms()),
        'heavy_atoms':len(heavy_keys),'all_atom_max_drift_A':max_drift,
        'residue_grouping_fallback':bool(omitted),'legacy_omitted_atom_indices':omitted,
        'raw_charge_sum_residues':sum(Chem.GetFormalCharge(r) for r in legacy.residues.values()),
        'prepared_charge_sum_residues':sum(Chem.GetFormalCharge(r) for r in prepared.residues.values()),
        'warnings':sorted(set(str(w.message) for w in notes)), 'residue_states':state}


def work(task):
    cid, out, max_draws = task
    case_dir = Path(out)/'cases'/cid
    case_dir.mkdir(parents=True,exist_ok=True)
    finished = case_dir/'result.json.gz'
    if finished.exists():
        return {'case_id':cid,'status':'cached','result':str(finished)}
    t0 = time.monotonic()
    rows = [r for r in csv.DictReader((E/'metrics.csv').open()) if r['case_id']==cid and r['variant']=='original']
    native_row = next(r for r in rows if r['kind']=='native')
    case = json.loads((E/f'cases/{cid}/case.json').read_text())
    old = json.loads((OLD/f'{cid}.json').read_text())
    old_rows = {(r['draw'],r['series']):r for r in old.get('rows',[]) if r.get('status')=='ok'}
    raw_prot, prep_prot, prep_audit = prepare_protein(case['pocket_path'])
    json_write(case_dir/'protein_preparation.json',prep_audit)
    fps = {method:plf.Fingerprint(DEFAULT_TYPES if method=='legacy' else TYPES, count=True,
                                implicit_hydrogens=method=='implicit_prepared') for method in METHODS}
    prot = {'legacy':raw_prot,'explicit_raw':raw_prot,'explicit_prepared':prep_prot,'implicit_prepared':prep_prot}
    native = read_mol(native_row['sdf'])
    template_maps = [a.GetAtomMapNum() for a in native.GetAtoms()]
    nat_h, nat_prep = add_hydrogens(native)
    ref = {}
    for method in METHODS:
        lig = nat_h if method.startswith('explicit') else native
        ev, pairs = events_of(lig, prot[method], fps[method])
        ref[method] = {'events':ev,'raw_pairs':pairs}
    expected_ref = next(iter(old_rows.values()))['ref_pairs'] if old_rows else None
    ref_parity = expected_ref is None or sorted(expected_ref)==ref['legacy']['raw_pairs']
    candidates = {(int(r['draw']),r['kind']):r for r in rows if r['kind'] in SERIES[:3]}
    rr_path = R/f'cases/{cid}/rows.json'
    refrows = {(r['draw'],r['arm']):r for r in json.loads(rr_path.read_text())} if rr_path.exists() else {}
    results = []
    with gzip.open(case_dir/'prepared_ligands.sdf.gz.tmp','wt') as sdf:
        nat_h.SetProp('_Name',f'{cid}_native')
        sdf.write(Chem.MolToMolBlock(nat_h)+'\n$$$$\n')
        for draw in range(max_draws):
            cp = E/f'draws/{cid}/original/core/{draw:02d}/condition.json'
            mask = json.loads(cp.read_text())['coordinate_fixed_mask'] if cp.exists() else None
            held_maps = [m for m,b in zip(template_maps,mask) if b] if mask is not None else None
            if mask is not None and len(mask)!=native.GetNumAtoms():
                raise ValueError('Coordinate mask size mismatch')
            for series in SERIES:
                row = {'draw':draw,'series':series,'held_maps':held_maps,'status':'unavailable_source'}
                if series in SERIES[:3]:
                    src = candidates.get((draw,series)); path = Path(src['sdf']) if src and src.get('sdf') else None
                    allowed = bool(src and src['status'] in ('ok','ready'))
                    row['source_status'] = src['status'] if src else 'missing'
                    row['pb_valid'] = src.get('pb_valid')=='True' if src else None
                    row['graph_preserved'] = src.get('graph_preserved')=='True' if src else None
                else:
                    src = refrows.get((draw,series)); path = R/f'cases/{cid}/{draw:02d}_{series}.sdf'
                    allowed = bool(src and src['status']=='ok')
                    row['source_status'] = src['status'] if src else 'missing'
                    row['pb_valid'] = src.get('pb_valid') if src else None
                    row['graph_preserved'] = src.get('graph_preserved') if src else None
                if not allowed or path is None or not path.exists() or held_maps is None:
                    results.append(row)
                    continue
                row.update(source=str(path),source_sha256=digest(path))
                try:
                    mol = normalize(read_mol(path),native)
                    row['atom_mapping_method'] = mol.GetProp('_IFP_atom_mapping_method')
                    mh, haudit = add_hydrogens(mol)
                    row['preparation'] = haudit
                    held_indices = [i for i,b in enumerate(mask) if b]
                    row['held_max_drift_to_native_A'] = float(np.linalg.norm(
                        mol.GetConformer().GetPositions()[held_indices]-native.GetConformer().GetPositions()[held_indices],axis=1).max()) if held_indices else 0.
                    mh.SetProp('_Name',f'{cid}_{series}_{draw:02d}')
                    sdf.write(Chem.MolToMolBlock(mh)+'\n$$$$\n')
                    row['methods'] = {}
                    for method in METHODS:
                        lig = mh if method.startswith('explicit') else mol
                        ev, pairs = events_of(lig,prot[method],fps[method])
                        row['methods'][method] = {'events':ev}
                        if method=='legacy':
                            old_row = old_rows.get((draw,series))
                            row['legacy_parity'] = None if old_row is None else pairs==sorted(old_row['cur_pairs'])
                    row['status'] = 'ok'
                except Exception as exc:
                    row.update(status='evaluation_failed',error=f'{type(exc).__name__}: {exc}')
                results.append(row)
    (case_dir/'prepared_ligands.sdf.gz.tmp').replace(case_dir/'prepared_ligands.sdf.gz')
    data = {'case_id':cid,'finished_at_kst':now(),'seconds':round(time.monotonic()-t0,3),
            'native_source':native_row['sdf'],'native_sha256':digest(native_row['sdf']),
            'native_H_preparation':nat_prep,'native_legacy_parity':ref_parity,'reference':ref,'rows':results}
    with gzip.open(case_dir/'result.json.gz.tmp','wt') as f:
        json.dump(data,f,separators=(',',':'),allow_nan=False)
    (case_dir/'result.json.gz.tmp').replace(finished)
    return {'case_id':cid,'status':'complete','seconds':data['seconds'],
            'ok_poses':sum(r['status']=='ok' for r in results),
            'failed_poses':sum(r['status']=='evaluation_failed' for r in results),
            'legacy_mismatches':sum(r.get('legacy_parity') is False for r in results)+int(not ref_parity)}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    study_path = out/'study.json'
    cids = sorted({r['case_id'] for r in csv.DictReader((E/'metrics.csv').open()) if r['variant']=='original' and r['kind']=='fm_core'})
    if args.case_ids:
        wanted = set(args.case_ids.split(','))
        cids = [c for c in cids if c in wanted]
        if len(cids)!=len(wanted):
            raise ValueError('Unknown case id requested')
    config = {'cases':cids,'draws_per_case':args.draws,'methods':METHODS,'series':SERIES,
              'source_metrics_sha256':digest(E/'metrics.csv'),'script_sha256':digest(__file__),
              'python':platform.python_version(),'prolif':plf.__version__,'rdkit':rdkit.__version__,
              'hydrogen_relaxation':'UFF 200 iterations; all ligand heavy atoms fixed; no pocket objective',
              'protonation':'Ligand encoded formal charge/H state retained per native graph; protein provided H coordinates retained; ProLIF amino-acid residue templates for prepared arms; no pKa/pH optimization',
              'vdw':'main event sets retain heavy-heavy VdW only; raw legacy residue pairs retained for parity',
              'threads_per_worker':1,'gpu':False}
    if study_path.exists():
        old = json.loads(study_path.read_text())
        if old['configuration']!=config:
            old_compatible = {k:v for k,v in old['configuration'].items() if k!='script_sha256'}
            new_compatible = {k:v for k,v in config.items() if k!='script_sha256'}
            if not args.repair_code or old_compatible!=new_compatible:
                raise ValueError('Run exists with different protocol/source/code; use another output')
            status_path = out/'status.json'
            if status_path.exists() and json.loads(status_path.read_text())['status']=='running':
                raise ValueError('Cannot repair while the original evaluation is running')
            revision = {'time_kst':now(),'reason':'Preserve disconnected fragments with the same PDB residue identity',
                        'previous_script_sha256':old['configuration']['script_sha256'],
                        'new_script_sha256':config['script_sha256'],
                        'cases_requiring_results':[c for c in cids if not (out/f'cases/{c}/result.json.gz').exists()]}
            old.setdefault('code_repairs',[]).append(revision)
            old['configuration']=config
            json_write(study_path,old)
            (out/f"script_repair_{config['script_sha256'][:12]}.py").write_text(Path(__file__).read_text())
            if (out/'execution_summary.json').exists():
                (out/'execution_summary_before_repair.json').write_text((out/'execution_summary.json').read_text())
    else:
        json_write(study_path,{'created_at_kst':now(),'configuration':config,'workers':args.workers,
                               'planned_pose_series_slots':len(cids)*args.draws*len(SERIES)})
        (out/'script_snapshot.py').write_text(Path(__file__).read_text())
    started = now()
    done,failed = [],[]
    status = {'status':'running','started_at_kst':started,'updated_at_kst':now(),
              'cases_total':len(cids),'cases_completed':0,'cases_failed':0,'workers':args.workers,
              'pid':os.getpid(),'gpu':False,'planned_pose_series_slots':len(cids)*args.draws*len(SERIES)}
    json_write(out/'status.json',status)
    emit(event='start',output=str(out),cases=len(cids),workers=args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work,(c,str(out),args.draws)):c for c in cids}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                info = future.result();done.append(info)
                emit(event='case_complete',**info)
            except Exception as exc:
                info = {'case_id':cid,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
                failed.append(info);emit(event='case_failed',case_id=cid,error=info['error'])
            status.update(updated_at_kst=now(),cases_completed=len(done),cases_failed=len(failed))
            json_write(out/'status.json',status)
    json_write(out/'execution_summary.json',{'finished_at_kst':now(),'completed':done,'failed':failed})
    status.update(status='evaluation_complete' if not failed else 'evaluation_partial_failure',finished_at_kst=now())
    json_write(out/'status.json',status)
    emit(event='evaluation_finished',**status)
    if failed:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--draws',type=int,default=32)
    ap.add_argument('--case-ids')
    ap.add_argument('--repair-code',action='store_true',help='Resume a finished run after a documented code-only repair')
    args = ap.parse_args()
    if not 1<=args.draws<=32 or args.workers<1:
        ap.error('draws must be 1..32 and workers positive')
    run(args)


if __name__=='__main__':
    main()
