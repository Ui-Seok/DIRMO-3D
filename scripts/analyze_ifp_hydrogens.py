"""Aggregate the CPU E5 hydrogen audit; no molecular calculations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np

SPEC = importlib.util.spec_from_file_location('ifp_eval',Path(__file__).with_name('reevaluate_ifp_hydrogens.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
GROUPS, METHODS, SERIES = module.GROUPS, module.METHODS, module.SERIES
REGIONS = ['all','core','free','mixed']
RESOLUTIONS = ['residue','atom']


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def average(values):
    finite = [v for v in values if v is not None]
    return statistics.mean(finite) if finite else None


def make_sets(events,held):
    out = {(resolution,region,group):set() for resolution in RESOLUTIONS for region in REGIONS for group in GROUPS}
    for ev in events:
        ids = set(ev['ligand_maps'])
        region = 'core' if ids <= held else ('free' if ids.isdisjoint(held) else 'mixed')
        pair = (ev['residue'],ev['type'])
        atom = (*pair,tuple(ev['ligand_maps']),tuple(ev['protein_atom_names']))
        for group,types in GROUPS.items():
            if ev['type'] not in types:
                continue
            for r in ['all',region]:
                out['residue',r,group].add(pair)
                out['atom',r,group].add(atom)
    return out


def compare(ref,cur):
    inter = len(ref & cur)
    return {'n_ref':len(ref),'n_cur':len(cur),'n_shared':inter,'n_new':len(cur-ref),
            'recovery':inter/len(ref) if ref else None,
            'new_rate':len(cur-ref)/len(cur) if cur else None,
            'tanimoto':inter/len(ref|cur) if ref or cur else None}


def bootstrap(a,b):
    keys = sorted(a.keys() & b.keys())
    delta = np.array([a[k]-b[k] for k in keys])
    if not len(delta):
        return {'cases':0,'delta':None,'ci99':None}
    rng = np.random.default_rng(20261023)
    reps = delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
    return {'cases':len(delta),'delta':float(delta.mean()),
            'ci99':[float(x) for x in np.quantile(reps,[.005,.995])],
            'resamples':5000,'seed':20261023}


def analyze(args):
    root,out = args.run.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    study = json.loads((root/'study.json').read_text())
    total_cases = len(study['configuration']['cases'])
    paths = sorted((root/'cases').glob('*/result.json.gz'))
    if len(paths)!=total_cases:
        raise ValueError(f'Result files {len(paths)} != planned {total_cases}')
    data = []
    for p in paths:
        with gzip.open(p,'rt') as f:
            data.append(json.load(f))
    counts = Counter(); errors = Counter(); statuses = Counter(); source_status = Counter()
    native_counts = {m:Counter() for m in METHODS}
    native_cases = {m:Counter() for m in METHODS}
    values = defaultdict(lambda:defaultdict(list))
    changed_proteins = 0
    incomplete_cases, grouping_repairs = set(),[]
    template_warning_counts = Counter()
    max_h_drift, max_protein_drift = 0.,0.
    common_slots,common_cases = 0,set()
    quality = {s:Counter() for s in SERIES}
    per_slot = []
    for case in data:
        cid = case['case_id']
        counts['legacy_native_mismatch'] += int(not case['native_legacy_parity'])
        counts['legacy_native_compared'] += int(any(r.get('legacy_parity') is not None for r in case['rows']))
        counts['native_H_UFF_nonconverged'] += int(case['native_H_preparation']['H_only_UFF_code']!=0)
        max_h_drift = max(max_h_drift,case['native_H_preparation']['heavy_max_drift_A'])
        protein = json.loads((root/f'cases/{cid}/protein_preparation.json').read_text())
        template_warning_counts.update(protein['warnings'])
        if any('different number of heavy atoms' in w for w in protein['warnings']):
            incomplete_cases.add(cid)
        if protein.get('residue_grouping_fallback'):
            grouping_repairs.append(cid)
        changed_proteins += int(protein['raw_charge_sum_residues']!=protein['prepared_charge_sum_residues'])
        max_protein_drift = max(max_protein_drift,protein['all_atom_max_drift_A'])
        for method in METHODS:
            evs = case['reference'][method]['events']
            native_counts[method].update(e['type'] for e in evs)
            native_cases[method].update(set(e['type'] for e in evs))
        draws = defaultdict(dict)
        for row in case['rows']:
            counts['planned_slots'] += 1
            statuses[row['status']] += 1
            source_status[f"{row['series']}|{row['source_status']}"] += 1
            if row.get('legacy_parity') is False:
                counts['legacy_pose_mismatch'] += 1
            if row.get('legacy_parity') is not None:
                counts['legacy_pose_compared'] += 1
            if row['status']=='evaluation_failed':
                errors[row['error']]+=1
            if row['status']=='ok':
                ha = row['preparation']
                counts['H_UFF_nonconverged'] += int(ha['H_only_UFF_code']!=0)
                max_h_drift = max(max_h_drift,ha['heavy_max_drift_A'])
                draws[row['draw']][row['series']]=row
        cache = {}
        for draw,series_rows in draws.items():
            if not all(s in series_rows and all(m in series_rows[s]['methods'] for m in METHODS) for s in SERIES):
                continue
            common_slots += 1;common_cases.add(cid)
            held = frozenset(series_rows['fm_core']['held_maps'])
            for series,row in series_rows.items():
                quality[series]['paired_slots'] += 1
                quality[series]['pb_pass'] += int(row['pb_valid'] is True)
                quality[series]['graph_stereo_pass'] += int(row['graph_preserved'] is True)
                quality[series]['held_pass'] += int(row['held_max_drift_to_native_A']<=.001)
                quality[series]['joint_pass'] += int(row['pb_valid'] is True and row['graph_preserved'] is True
                                                   and row['held_max_drift_to_native_A']<=.001)
            for method in METHODS:
                if (method,held) not in cache:
                    cache[method,held] = make_sets(case['reference'][method]['events'],held)
                reference = cache[method,held]
                for series,row in series_rows.items():
                    current = make_sets(row['methods'][method]['events'],held)
                    for selection, ref in reference.items():
                        metrics = compare(ref,current[selection])
                        key = (method,*selection,series)
                        values[key][cid].append(metrics)
                        # Preserve detailed primary metrics without a huge all-metrics file.
                        if method in ['explicit_prepared','implicit_prepared'] and selection[0]=='atom' and selection[1] in ['core','free'] and selection[2] in ['polar_pi','hydrogen_bond']:
                            per_slot.append({'case_id':cid,'draw':draw,'series':series,'method':method,
                                             'resolution':selection[0],'region':selection[1],'group':selection[2],**metrics})

    rows,by_key,case_metric = [],{},{}
    for key,cases in sorted(values.items()):
        method,resolution,region,group,series = key
        raw = [r for rs in cases.values() for r in rs]
        result = dict(zip(['method','resolution','region','group','series'],key))
        result.update(paired_slots=len(raw),paired_cases=len(cases))
        for metric in ['n_ref','n_cur','n_shared','n_new','recovery','new_rate','tanimoto']:
            means = {c:average(r[metric] for r in rs) for c,rs in cases.items()}
            means = {c:v for c,v in means.items() if v is not None}
            result[f'{metric}_slot_mean'] = average(r[metric] for r in raw)
            result[f'{metric}_case_mean'] = average(means.values())
            result[f'{metric}_cases'] = len(means)
            result[f'{metric}_slots'] = sum(r[metric] is not None for r in raw)
            clean = {c:v for c,v in means.items() if c not in incomplete_cases}
            result[f'{metric}_exclude_incomplete_case_mean'] = average(clean.values())
            result[f'{metric}_exclude_incomplete_cases'] = len(clean)
            if metric in ['recovery','new_rate','tanimoto']:
                case_metric[key,metric] = means
        rows.append(result);by_key[key]=result
    if not rows:
        raise ValueError('No matched candidate sets')
    with (out/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with gzip.open(out/'primary_per_slot.csv.gz','wt') as f:
        w=csv.DictWriter(f,fieldnames=list(per_slot[0]));w.writeheader();w.writerows(per_slot)
    # Exploratory comparisons: choose the same primary endpoints for both H approaches.
    selections = [('atom',r,g) for r in ['core','free'] for g in ['polar_pi','hydrogen_bond']]
    selections += [('residue','all','common_non_vdw')]
    ci = []
    for method in ['explicit_prepared','implicit_prepared']:
        for selection in selections:
            for series in ['fm_whole','uff_lig','uff_pkt','uff_pkt_soft','uff_pkt_tether']:
                for metric in ['recovery','new_rate','tanimoto']:
                    result = bootstrap(case_metric[(method,*selection,series),metric],
                                       case_metric[(method,*selection,'fm_core'),metric])
                    ci.append({'method':method,'resolution':selection[0],'region':selection[1],
                               'group':selection[2],'series':series,'control':'fm_core',
                               'metric':metric,'sensitivity':'all_cases',**result})
    for method in ['explicit_prepared','implicit_prepared']:
        for region in ['core','free']:
            for series in ['fm_whole','uff_pkt','uff_pkt_tether']:
                current = case_metric[(method,'atom',region,'polar_pi',series),'recovery']
                control = case_metric[(method,'atom',region,'polar_pi','fm_core'),'recovery']
                result = bootstrap({c:v for c,v in current.items() if c not in incomplete_cases},
                                   {c:v for c,v in control.items() if c not in incomplete_cases})
                ci.append({'method':method,'resolution':'atom','region':region,'group':'polar_pi',
                           'series':series,'control':'fm_core','metric':'recovery',
                           'sensitivity':'exclude_incomplete_protein_residues',**result})
    write_json(out/'paired_bootstrap.json',ci)
    summary = {'finished_at_kst':now(),'run':str(root),'cases_processed':len(data),
               'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'study_sha256':hashlib.sha256((root/'study.json').read_bytes()).hexdigest(),
               'numpy_version':np.__version__,
               'planned_pose_series_slots':study['planned_pose_series_slots'],
               'pose_status_counts':dict(statuses),'source_status_counts':dict(source_status),
               'evaluation_error_reasons':dict(errors),'audit_counts':dict(counts),
               'common_seven_series_draw_slots':common_slots,'common_cases':len(common_cases),
               'paired_saved_quality_counts':{s:dict(v) for s,v in quality.items()},
               'prepared_protein_charge_changed_cases':changed_proteins,
               'incomplete_protein_residue_cases':sorted(incomplete_cases),
               'residue_grouping_repaired_cases':grouping_repairs,
               'protein_template_warning_counts':dict(template_warning_counts),
               'max_ligand_heavy_coordinate_change_A':max_h_drift,
               'max_protein_coordinate_change_A':max_protein_drift,
               'native_atom_contact_counts':{m:dict(v) for m,v in native_counts.items()},
               'native_cases_with_contact_type':{m:dict(v) for m,v in native_cases.items()},
               'comparisons':'Exploratory case-clustered bootstrap; 99% intervals, no familywise correction',
               'primary_rows':[r for r in rows if r['method'] in ['explicit_prepared','implicit_prepared'] and
                               (r['resolution'],r['region'],r['group']) in selections]}
    write_json(out/'summary.json',summary)
    plot(out,by_key,native_counts)
    if counts['legacy_native_mismatch'] or counts['legacy_pose_mismatch']:
        raise ValueError('Legacy parity gate failed; results require investigation')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['primary_rows','native_atom_contact_counts','source_status_counts']},ensure_ascii=False,indent=2))


def plot(out,rows,native_counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Prior','FM core','FM whole','Lig. UFF','Pocket UFF','Soft core','Tether']
    fig,axes=plt.subplots(1,3,figsize=(16,5.3),layout='constrained')
    choices=[('residue','all','common_non_vdw','All-region contact recovery\n(residue, common non-VdW)'),
             ('atom','core','polar_pi','Protected core contact recovery\n(atom, polar / pi)'),
             ('atom','free','polar_pi','Free-region contact recovery\n(atom, polar / pi)')]
    for ax,(resolution,region,group,title) in zip(axes,choices):
        x=np.arange(len(SERIES))
        for offset,(method,color,label) in zip([-.19,.19],[('explicit_prepared','#26775e','Explicit H, prepared protein'),('implicit_prepared','#6e91bf','Implicit H, prepared protein')]):
            ys=[rows[(method,resolution,region,group,s)]['recovery_case_mean'] for s in SERIES]
            ys=[100*y if y is not None else np.nan for y in ys]
            ax.bar(x+offset,ys,.36,color=color,label=label)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,105),title=title,ylabel='Case-average recovery (%)')
        ax.tick_params(axis='x',labelrotation=50)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.supxlabel('Matched draws across all seven series. Native reference contacts differ between H methods; compare treatments within each method.\nHeavy coordinates unchanged. Error intervals and valid case counts are in the report / paired_bootstrap.json. 2026-10-23 KST.',fontsize=9)
    fig.savefig(out/'recovery.png',dpi=180);fig.savefig(out/'recovery.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    types=['HBDonor','HBAcceptor','Anionic','Cationic','PiStacking','CationPi','PiCation']
    x=np.arange(len(types))
    for offset,(method,color,label) in zip([-.25,0,.25],[('legacy','#aaaaaa','Legacy (missing ligand H)'),('explicit_prepared','#26775e','Explicit H, prepared protein'),('implicit_prepared','#6e91bf','Implicit H, prepared protein')]):
        vals=[native_counts[method][t] for t in types]
        bars=ax.bar(x+offset,vals,.24,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=types,ylabel='Native atom-contact count',title='Native references: interaction coverage after hydrogen preparation')
    ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'native_contact_coverage.png',dpi=180);fig.savefig(out/'native_contact_coverage.svg');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    analyze(ap.parse_args())
