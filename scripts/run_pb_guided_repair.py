"""Run raw generation -> exact PB diagnosis -> targeted pose repair on GPU4.

Initial generation and CPU diagnostics overlap. One repair GPU worker starts
after initial GPU generation, then the previously paused GPU workers resume.
Every input, including build failures and unlocalizable PB failures, is kept.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dirmo.chemistry.graphs import read_sdf
from dirmo.evaluation.pb_guided import noisy_input, preservation, select_regions, uff_control
from dirmo.evaluation.pb_worker import bounded_measure as measure
from dirmo.gpu import inventory, resolve_device, worker_environment
from dirmo.gpu_priority import set_saved_workers_paused
from dirmo.io import append_jsonl, digest, sha256_file, write_json
from generate_pb_initial import read

ARMS = ('targeted', 'random', 'whole', 'UFF_targeted')
PAUSED = ROOT / 'runs/fm_pose_gpu4_2000_batch32_v1/steps_256/gpu_pause.json'


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, TypeError, ValueError):
        return False


def arm_directory(root, arm, case_id, draw, candidate):
    return root / 'repairs' / arm / case_id / f'{draw:03d}' / f'{candidate:02d}'


def evaluate_initial(root, case_id):
    plan = read(root / 'study.json')
    case_index, case = next((i, c) for i, c in enumerate(plan['cases']) if c['case_id'] == case_id)
    result_file = root / 'evaluations' / 'initial' / f'{case_id}.json'
    result = read(result_file, {'case_id': case_id, 'draws': {}})
    sources = []
    for draw in range(plan['initial_draws']):
        key = f'{draw:03d}'
        directory = root / 'cases' / case_id / 'initial' / key
        generated = read(directory / 'result.json', {})
        if key not in result['draws']:
            pb = measure(generated['raw_sdf'], case['pocket_path'], diagnose=True) if generated.get('raw_sdf') else {
                'passes': False, 'missing': True, 'checks': {}, 'reason': generated.get('reason', generated.get('status', 'missing'))}
            row = {'draw': draw, 'generation_status': generated.get('status', 'missing'), 'pb': pb,
                   'repair_eligible': False}
            if not pb['passes'] and not pb.get('missing') and generated.get('repair_input_sdf'):
                mol = read_sdf(generated['repair_input_sdf'])
                mask = select_regions(mol, pb.get('events', []), seed=plan['seed'] + case_index * 100 + draw)
                row['mask'] = mask
                row['repair_eligible'] = mask['eligible']
            else:
                row['ineligible_reason'] = 'Already PB valid' if pb['passes'] else 'Missing measurement or no connected input graph for pose-only repair'
            result['draws'][key] = row
            write_json(result_file, result)
        row = result['draws'][key]
        if not row['repair_eligible']:
            continue
        source_directory = root / 'cases' / case_id / 'sources' / key
        source_file = source_directory / 'source.json'
        if source_file.exists():
            sources.append(str(source_file))
            continue
        original = generated['repair_input_sdf']
        mol = read_sdf(original)
        source = {'case_id': case_id, 'draw': draw, 'original_sdf': original,
                  'original_sha256': sha256_file(original), 'pocket_path': case['pocket_path'],
                  'protein_path': case['protein_path'], 'pb_before': row['pb'], 'mask': row['mask'], 'items': {}}
        for arm in ('targeted', 'random', 'whole'):
            free = row['mask'][f'{arm}_indices']
            maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in free]
            items = []
            for candidate in range(plan['repair_draws']):
                destination = arm_directory(root, arm, case_id, draw, candidate)
                seed = plan['seed'] + 7000000 + case_index * 100000 + draw * 100 + candidate
                input_path = source_directory / 'priors' / arm / f'{candidate:02d}.sdf'
                try:
                    prior = noisy_input(original, free, seed, input_path)
                    item = {'case_id': f'{case_id}/{key}', 'seed': seed, 'free_maps': maps,
                            'pocket_path': case['pocket_path'], 'input_sdf': str(input_path),
                            'output_dir': str(destination), 'prior': prior}
                    items.append(item)
                except Exception as exc:
                    write_json(destination / 'result.json', {'status': 'initialization_failed', 'reason': f'{type(exc).__name__}: {exc}'})
            source['items'][arm] = items
            source.setdefault('distinct_prior_xyz', {})[arm] = len({digest(read_sdf(item['input_sdf']).GetConformer().GetPositions().tolist()) for item in items})
        uff_path = arm_directory(root, 'UFF_targeted', case_id, draw, 0)
        try:
            control = uff_control(original, row['mask']['targeted_indices'], uff_path / 'raw.sdf')
            write_json(uff_path / 'result.json', {'status': 'ok', 'sdf_path': str(uff_path / 'raw.sdf'), **control})
        except Exception as exc:
            write_json(uff_path / 'result.json', {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
        write_json(source_file, source)
        sources.append(str(source_file))
    result['complete'] = True
    write_json(result_file, result)
    write_json(root / 'cases' / case_id / 'requests_ready.json', {'case_id': case_id, 'sources': sources})


def gpu_repair(root, gpu):
    from dirmo.config import FMConfig
    from dirmo.evaluation.fm_quality import CHECKPOINTS
    from dirmo.generation.pose_repair import PoseRepairBackend
    plan = read(root / 'study.json')
    device = resolve_device(gpu, inventory())
    os.environ.update(worker_environment(device, str(ROOT / 'src')))
    filename, checksum = CHECKPOINTS['base']
    config = FMConfig(gpu=gpu, port=8876, python=sys.executable, repo_path='/home/uslee/code/flowr_root',
                      checkpoint_path=str(Path('/LocalData1/uslee/checkpoints/FLOWR_root') / filename),
                      checkpoint_sha256=checksum)
    state = {'status': 'loading', 'pid': os.getpid(), 'gpu': device}
    write_json(root / 'repair_worker.json', state)

    def acquire_gpu():
        record = read(PAUSED, {})
        if record.get('status') == 'gpu_paused':
            set_saved_workers_paused(PAUSED, True, root, 'Verify saved GPU4 identities before priority repair')
            return
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
                       '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used <= 35000:
                set_saved_workers_paused(PAUSED, True, root, 'PB-guided repair inputs ready')
                after = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
                            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                if after <= 38000:
                    return
                set_saved_workers_paused(PAUSED, False, root, 'Wait for an original-job batch with lower live memory')
            state.update(status='waiting_for_gpu_memory')
            write_json(root / 'repair_worker.json', state)
            time.sleep(1)

    acquire_gpu()
    backend = PoseRepairBackend(config, allow_whole_pose=True)
    state['kernel'] = backend.identity
    pending_cases = [case for case in plan['cases'] if not
                     (root / 'cases' / case['case_id'] / 'repair_generation_complete.json').exists()]
    while pending_cases:
        ready = [case for case in pending_cases if (root / 'cases' / case['case_id'] / 'requests_ready.json').exists()]
        if not ready:
            backend.torch.cuda.empty_cache()
            set_saved_workers_paused(PAUSED, False, root, 'Priority repair waiting for CPU PB diagnosis')
            state.update(status='waiting_for_diagnosis', remaining_cases=len(pending_cases))
            write_json(root / 'repair_worker.json', state)
            time.sleep(3)
            continue
        # Process ready cases without waiting for a slow earlier CPU evaluation.
        # The cohort remains fixed; this changes execution order, not selection.
        case = ready[0]
        case_id = case['case_id']
        ready_file = root / 'cases' / case_id / 'requests_ready.json'
        if read(ready_file)['sources']:
            acquire_gpu()
        for source_file in read(ready_file)['sources']:
            source = read(source_file)
            for arm in ('targeted', 'random', 'whole'):
                pending = [item for item in source['items'][arm] if not (Path(item['output_dir']) / 'result.json').exists()]
                capacity = min(8, len(pending))
                while pending:
                    batch = pending[:capacity]
                    state.update(status='running', current_case=case_id, initial_draw=source['draw'], arm=arm)
                    write_json(root / 'repair_worker.json', state)
                    try:
                        _, stats = backend.generate(batch, plan['steps'])
                        append_jsonl(root / 'repair_batches.jsonl', {'case_id': case_id, 'initial_draw': source['draw'], 'arm': arm, **stats})
                        pending = pending[len(batch):]
                    except backend.torch.OutOfMemoryError:
                        backend.torch.cuda.empty_cache()
                        append_jsonl(root / 'repair_batches.jsonl', {'status': 'oom', 'case_id': case_id, 'arm': arm,
                                     'batch_size': len(batch), 'partial_forward_calls': backend.last_forward_calls})
                        if capacity > 1:
                            capacity = max(1, capacity // 2)
                            continue
                        write_json(Path(batch[0]['output_dir']) / 'result.json', {'status': 'oom'})
                        pending = pending[1:]
                    except Exception as exc:
                        traceback.print_exc()
                        for item in batch:
                            write_json(Path(item['output_dir']) / 'result.json', {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
                        pending = pending[len(batch):]
                    backend.torch.cuda.empty_cache()
        write_json(root / 'cases' / case_id / 'repair_generation_complete.json', {'case_id': case_id})
        pending_cases.remove(case)
    state.update(status='complete')
    write_json(root / 'repair_worker.json', state)
    set_saved_workers_paused(PAUSED, False, root, 'Priority GPU repair complete; CPU evaluation continues')


def evaluate_repairs(root, case_id):
    plan = read(root / 'study.json')
    source_files = read(root / 'cases' / case_id / 'requests_ready.json')['sources']
    result_file = root / 'evaluations' / 'repairs' / f'{case_id}.json'
    result = read(result_file, {'case_id': case_id, 'sources': {}})
    for source_file in source_files:
        source = read(source_file)
        key = f"{source['draw']:03d}"
        result['sources'].setdefault(key, {})
        for arm in ARMS:
            if arm in result['sources'][key]:
                continue
            free = source['mask'][f"{'targeted' if arm == 'UFF_targeted' else arm}_indices"]
            candidates = []
            for candidate in range(1 if arm == 'UFF_targeted' else plan['repair_draws']):
                directory = arm_directory(root, arm, case_id, source['draw'], candidate)
                draw = read(directory / 'result.json', {})
                path = draw.get('sdf_path')
                metrics = measure(path, source['pocket_path']) if path else {'passes': False, 'missing': True, 'checks': {}}
                contract = preservation(source['original_sdf'], path, free) if path else {'graph_preserved': False, 'held_preserved': False}
                prior_path = root / 'cases' / case_id / 'sources' / key / 'priors' / arm / f'{candidate:02d}.sdf'
                noise_control = None
                if arm != 'UFF_targeted' and prior_path.exists():
                    prior_metrics = measure(prior_path, source['pocket_path'])
                    prior_contract = preservation(source['original_sdf'], prior_path, free)
                    noise_control = {'pb': prior_metrics, 'contract': prior_contract,
                                     'success': bool(prior_metrics['passes'] and prior_contract['graph_preserved'] and prior_contract['held_preserved'])}
                candidates.append({'candidate': candidate, 'status': draw.get('status', 'missing'), 'sdf': path,
                                   'pb': metrics, 'contract': contract, 'noise_only_control': noise_control,
                                   'success': bool(metrics['passes'] and contract['graph_preserved'] and contract['held_preserved'])})
            selected = min(candidates, key=lambda row: (not row['success'], len(row['pb'].get('failed_checks', [])) + 20 * bool(row['pb'].get('missing')),
                                                       row['contract'].get('rmsd_to_initial_generated_pose', float('inf')), row['candidate']))
            old_failures = set(source['pb_before']['failed_checks'])
            result['sources'][key][arm] = {'success': any(row['success'] for row in candidates),
                    'selected_candidate': selected['candidate'], 'selected_sdf': selected['sdf'],
                    'resolved_checks': sorted(old_failures - set(selected['pb'].get('failed_checks', []))) if not selected['pb'].get('missing') else [],
                    'new_failed_checks': sorted(set(selected['pb'].get('failed_checks', [])) - old_failures),
                    'candidates': candidates, 'selection': 'PB+contract success first; fewest failed checks; smallest displacement from generated input; index tie-break'}
            write_json(result_file, result)
    result['complete'] = True
    write_json(result_file, result)


def seal_cpu_failure(root, phase, case_id, reason):
    """Keep a failed CPU task explicit while allowing other cases to continue."""
    plan = read(root / 'study.json')
    if phase == 'initial':
        path = root / 'evaluations' / 'initial' / f'{case_id}.json'
        result = read(path, {'case_id': case_id, 'draws': {}})
        for draw in range(plan['initial_draws']):
            result['draws'].setdefault(f'{draw:03d}', {'draw': draw, 'generation_status': 'evaluation_unavailable',
                 'repair_eligible': False, 'pb': {'passes': False, 'missing': True, 'checks': {}, 'error': reason}})
        result.update(complete=True, evaluation_incomplete=True, error=reason)
        write_json(path, result)
        sources = sorted(str(p) for p in (root / 'cases' / case_id / 'sources').glob('*/source.json'))
        write_json(root / 'cases' / case_id / 'requests_ready.json',
                   {'case_id': case_id, 'sources': sources, 'evaluation_incomplete': True, 'error': reason})
    else:
        path = root / 'evaluations' / 'repairs' / f'{case_id}.json'
        result = read(path, {'case_id': case_id, 'sources': {}})
        for source_file in read(root / 'cases' / case_id / 'requests_ready.json')['sources']:
            key = f"{read(source_file)['draw']:03d}"
            arms = result['sources'].setdefault(key, {})
            for arm in ARMS:
                if arm in arms:
                    continue
                candidates = [{'candidate': index, 'status': 'evaluation_unavailable', 'sdf': None,
                    'success': False, 'pb': {'passes': False, 'missing': True, 'checks': {}, 'error': reason},
                    'contract': {'graph_preserved': False, 'held_preserved': False}}
                    for index in range(1 if arm == 'UFF_targeted' else plan['repair_draws'])]
                arms[arm] = {'success': False, 'selected_candidate': None, 'selected_sdf': None,
                             'candidates': candidates, 'evaluation_incomplete': True, 'error': reason}
        result.update(complete=True, evaluation_incomplete=True, error=reason)
        write_json(path, result)
    append_jsonl(root / 'evaluation_failures.jsonl', {'phase': phase, 'case_id': case_id, 'reason': reason})


def snapshot(root, status):
    plan = read(root / 'study.json')
    counts = Counter()
    arm_counts = {arm: Counter() for arm in ARMS}
    per_case = []
    for case in plan['cases']:
        initial = read(root / 'evaluations' / 'initial' / f"{case['case_id']}.json", {})
        repairs = read(root / 'evaluations' / 'repairs' / f"{case['case_id']}.json", {})
        counts['initial_cases_complete'] += bool(initial.get('complete'))
        counts['repair_cases_complete'] += bool(repairs.get('complete'))
        counts['case_evaluation_incomplete'] += bool(initial.get('evaluation_incomplete') or repairs.get('evaluation_incomplete'))
        counts['initial_generation_cases_complete'] += (root / 'cases' / case['case_id'] / 'initial' / 'complete.json').exists()
        case_row = {'case_id': case['case_id'], 'eligible': 0, **{arm: 0 for arm in ARMS}}
        for key, row in initial.get('draws', {}).items():
            counts['initial_evaluated'] += 1
            counts['initial_pb_valid'] += bool(row['pb']['passes'])
            counts['initial_missing'] += bool(row['pb'].get('missing'))
            counts['initial_measurement_timeout'] += bool(row['pb'].get('measurement_timeout'))
            counts['initial_evaluation_unknown'] += bool(row['pb'].get('missing') and row.get('generation_status') in ('built', 'evaluation_unavailable'))
            counts['initial_pb_failed_measured'] += not row['pb']['passes'] and not row['pb'].get('missing')
            counts['repair_eligible'] += bool(row['repair_eligible'])
            case_row['eligible'] += bool(row['repair_eligible'])
            for arm, evaluation in repairs.get('sources', {}).get(key, {}).items():
                arm_counts[arm]['evaluated_sources'] += 1
                arm_counts[arm]['success_within_budget'] += bool(evaluation['success'])
                arm_counts[arm]['scheduled_attempts_evaluated'] += len(evaluation['candidates'])
                arm_counts[arm]['successful_attempts'] += sum(c['success'] for c in evaluation['candidates'])
                counts['repair_measurement_unknown'] += sum(bool(c['pb'].get('missing')) and c['status'] not in
                    ('build_failed', 'failed', 'oom', 'initialization_failed') for c in evaluation['candidates'])
                case_row[arm] += bool(evaluation['success'])
        per_case.append(case_row)
    if status == 'complete' and (counts['case_evaluation_incomplete'] or counts['initial_evaluation_unknown'] or counts['repair_measurement_unknown']):
        status = 'complete_with_missing_evaluation'
    summary = {'status': status, 'updated_utc': datetime.now(timezone.utc).isoformat(),
               'planned_initial_draws': len(plan['cases']) * plan['initial_draws'],
               'counts': dict(counts), 'arms': {arm: dict(counter) for arm, counter in arm_counts.items()},
               'initial_worker': read(root / 'initial_worker.json', {}),
               'repair_worker': read(root / 'repair_worker.json', {}), 'per_case': per_case,
               'paused_256': read(PAUSED, {}).get('status')}
    write_json(root / 'summary.json', summary)
    lines = ['# FLOWR 생성물의 PB 진단 기반 부분 pose 재생성', '', f'상태: {status}. 갱신(UTC): {summary["updated_utc"]}', '',
             f"예정: 100 pockets × 16 initial draws = {summary['planned_initial_draws']:,}. 64-step base checkpoint. 추가 학습·LLM 호출 없음.", '',
             '초기 생성은 graph/좌표를 고정하지 않는 pocket-only 생성이다. Pocket 위치·원자 수는 기존 복합체에서 정하며 공식 SPINDR 논문 재현 실험은 아니다.', '',
             '재생성은 화학 그래프와 입체화학을 유지하는 pose-only 실험이다. 자유 좌표에 0.3 Å 노이즈를 주고 나머지 좌표는 hard hold한다. 각 FM 방법은 입력당 최대 8개 후보, 1회 수정이다.', '',
             '문제 위치는 동일 PB dock 설정의 실제 원자쌍·결합·각도·평면성 위반에서 정한다. 내부 에너지 등 전역 실패를 임의의 fragment 점수로 분해하지 않는다. 위치를 정할 수 없는 실패와 생성 실패도 전체 초기 분모에 유지한다.', '',
             'Random control은 선택된 영역과 원자 수·경계 결합 수가 정확하게 같은 연결 영역에서 무작위로 고른다. 동일 영역을 고른 경우 및 대안이 없는 경우도 기록한다.', '',
             f"초기 GPU 생성 완료 pocket: {counts['initial_generation_cases_complete']}/100; 초기 PB 측정: {counts['initial_evaluated']}/{summary['planned_initial_draws']}; PB 통과: {counts['initial_pb_valid']}; 측정된 PB 실패: {counts['initial_pb_failed_measured']}; 누락/생성 실패: {counts['initial_missing']}; 부분 복구 대상: {counts['repair_eligible']}.", '',
             '| 방법 | 복구 평가 입력 | 후보 예산 내 성공 입력 | 평가된 생성 시도 |', '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        counter = arm_counts[arm]
        lines.append(f"| {arm} | {counter['evaluated_sources']} | {counter['success_within_budget']} | {counter['scheduled_attempts_evaluated']} |")
    lines += ['', '성공 = 기존 생성물의 graph/stereo 보존 + held drift ≤0.001 Å + 고정 6 Å pocket의 PoseBusters dock 전체 검사 통과. 실험 정답 pose가 없는 새 분자이므로 native RMSD 개선을 요구하지 않는다.', '',
              'UFF_targeted는 같은 진단 영역의 좌표만 풀고 ligand-only UFF를 최대 200회 적용한 별도 CPU 대조군이다. 8회 FM 생성과 계산 비용이 같다고 간주하지 않는다.', '',
              'PB를 직접 최적화한 결과이므로 binding affinity 또는 독립적인 물리 검증 개선을 주장하지 않는다. 전체 protein/water/cofactor 검사는 이번 주 지표에 포함하지 않는다.', '',
              f"초기 PB 평가 시간 초과: {counts['initial_measurement_timeout']}; 초기 평가 미확정: {counts['initial_evaluation_unknown']}; 복구 평가 미확정: {counts['repair_measurement_unknown']}. 이들은 물리 통과나 측정된 물리 실패로 바꾸지 않는다.", '',
              'GPU4의 256-step은 PB 입력을 기다리는 동안 실행하고, 복구 입력이 준비되면 PID/명령을 검증해 잠시 정지한 뒤 새 실험에 우선권을 준다. 공용 scheduler와 GPU3 작업은 유지한다. PB의 50-conformer 수·검사·임계값은 유지하고 개별 측정을 별도 CPU 프로세스로 격리한다. 600초 시간 초과는 미측정으로 기록하며 전체 제어기를 중단하지 않는다.', '',
              '로그: initial.log, repair.log, controller.log, evaluations/logs/*.log. 입력별 PB 근거·mask·초기 노이즈는 cases/*/sources/*/source.json에, 원본 raw 생성물과 재생성물은 cases/*/initial 및 repairs/에 보관한다.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return summary


def resume_previous(root):
    return set_saved_workers_paused(PAUSED, False, root, 'Priority GPU stage finished or awaiting CPU evaluation')


def finalize(root):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = read(root / 'summary.json')
    initial_rows, repair_rows, candidate_rows = [], [], []
    plan = read(root / 'study.json')
    for case in plan['cases']:
        cid = case['case_id']
        initial = read(root / 'evaluations' / 'initial' / f'{cid}.json')
        repaired = read(root / 'evaluations' / 'repairs' / f'{cid}.json')
        for key, row in initial['draws'].items():
            initial_rows.append({'case_id': cid, 'draw': int(key), 'pb_valid': row['pb']['passes'],
                                 'missing': bool(row['pb'].get('missing')), 'repair_eligible': row['repair_eligible'],
                                 'failed_checks': ';'.join(row['pb'].get('failed_checks', []))})
        for key, arms in repaired['sources'].items():
            for arm, row in arms.items():
                repair_rows.append({'case_id': cid, 'draw': int(key), 'arm': arm, 'success': row['success'],
                                    'selected_candidate': row['selected_candidate'], 'selected_sdf': row['selected_sdf']})
                for candidate in row['candidates']:
                    candidate_rows.append({'case_id': cid, 'draw': int(key), 'arm': arm,
                        'candidate': candidate['candidate'], 'status': candidate['status'], 'success': candidate['success'],
                        'pb_valid': candidate['pb']['passes'], **candidate['contract'], 'sdf': candidate['sdf']})
    for name, rows in (('initial_metrics', initial_rows), ('repair_metrics', repair_rows), ('candidate_metrics', candidate_rows)):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (root / f'{name}.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
    eligible = summary['counts'].get('repair_eligible', 0)
    total = summary['planned_initial_draws']
    valid = summary['counts'].get('initial_pb_valid', 0)
    statistics = {'initial_pb_valid_pct': 100 * valid / total, 'repair_eligible': eligible, 'arms': {}}
    for arm in ARMS:
        counter = summary['arms'][arm]
        successes = counter.get('success_within_budget', 0)
        statistics['arms'][arm] = {'repaired_of_eligible_pct': 100 * successes / eligible if eligible else None,
                                  'end_to_end_pb_and_contract_valid_pct': 100 * (valid + successes) / total}
    if eligible:
        rng = np.random.default_rng(2042)
        matrix = np.array([[r['eligible'], r['targeted'], r['random'], r['whole']] for r in summary['per_case']])
        differences = {'targeted_minus_random_pp': [], 'targeted_minus_whole_pp': []}
        for _ in range(1000):
            sampled = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
            if sampled[0]:
                differences['targeted_minus_random_pp'].append(100 * (sampled[1] - sampled[2]) / sampled[0])
                differences['targeted_minus_whole_pp'].append(100 * (sampled[1] - sampled[3]) / sampled[0])
        statistics['pocket_cluster_bootstrap_95'] = {key: np.quantile(values, [.025, .975]).tolist() for key, values in differences.items() if values}
    write_json(root / 'statistics.json', statistics)
    labels = ['PB-guided', 'Random region', 'Whole pose', 'UFF targeted']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, values, title in ((axes[0], [statistics['arms'][a]['repaired_of_eligible_pct'] or 0 for a in ARMS],
                             f'Failed inputs repaired within budget (n={eligible})'),
                            (axes[1], [statistics['arms'][a]['end_to_end_pb_and_contract_valid_pct'] for a in ARMS],
                             f'Final valid output per initial draw (n={total})')):
        bars = ax.bar(labels, values, color=['#2878A5', '#ADB5BD', '#7A5195', '#D77D29'])
        ax.bar_label(bars, fmt='%.1f', padding=3)
        ax.set(title=title, ylabel='Percent', ylim=(0, 105))
        ax.tick_params(axis='x', rotation=12)
    axes[1].axhline(statistics['initial_pb_valid_pct'], linestyle='--', color='#444444', label='Initial PB-valid')
    axes[1].legend(frameon=False)
    fig.suptitle('FLOWR base: PB-guided pose repair / 64 steps / 8 FM candidates')
    fig.tight_layout()
    fig.savefig(root / 'results.png', dpi=180); fig.savefig(root / 'results.pdf')
    plt.close(fig)
    with (root / 'REPORT.md').open('a') as handle:
        handle.write('\n## 최종 집계\n\n' + json.dumps(statistics, indent=2, ensure_ascii=False) + '\n\n![결과](results.png)\n')


def run(root, gpu):
    plan = read(root / 'study.json')
    cpu_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLCONFIGDIR': '/tmp/dirmo_pb_mpl',
               'DIRMO_PB_ENERGY_CACHE': str(root / 'cache' / 'reference_energy'),
               'DIRMO_PB_MEASURE_CACHE': str(root / 'cache' / 'measurements'), 'DIRMO_PB_MEASUREMENT_TIMEOUT': '600'}
    streams, tasks, repair_process = [], {}, None

    def cpu_task(phase, cid):
        log = root / 'evaluations' / 'logs' / f'{phase}_{cid}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, '--output', str(root), '--phase', phase, '--case-id', cid]
        reason = ''
        for attempt in range(2):
            try:
                with log.open('a') as stream:
                    process = subprocess.run(command, cwd=ROOT, env=cpu_env, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=18000)
                if not process.returncode:
                    return
                reason = f'CPU {phase} exited with {process.returncode}; see {log}'
            except subprocess.TimeoutExpired:
                reason = f'CPU {phase} case exceeded 18000 seconds; see {log}'
            append_jsonl(root / 'cpu_retries.jsonl', {'phase': phase, 'case_id': cid, 'attempt': attempt + 1, 'reason': reason})
        seal_cpu_failure(root, phase, cid, reason)

    write_json(root / 'controller.json', {'status': 'running', 'pid': os.getpid(), 'gpu': gpu})
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        while True:
            initial_state = read(root / 'initial_worker.json', {})
            if initial_state.get('status') != 'complete' and not alive(initial_state.get('pid')):
                raise RuntimeError('Initial GPU worker stopped before completion')
            for case in plan['cases']:
                cid = case['case_id']
                for phase, marker, evaluation in (
                    ('initial', root / 'cases' / cid / 'initial' / 'complete.json', root / 'cases' / cid / 'requests_ready.json'),
                    ('evaluate', root / 'cases' / cid / 'repair_generation_complete.json', root / 'evaluations' / 'repairs' / f'{cid}.json')):
                    key = (phase, cid)
                    complete = evaluation.exists() if phase == 'initial' else bool(read(evaluation, {}).get('complete'))
                    if marker.exists() and not complete and key not in tasks:
                        tasks[key] = pool.submit(cpu_task, phase, cid)
            for future in tasks.values():
                if future.done():
                    future.result()
            repair_state = read(root / 'repair_worker.json', {})
            if initial_state.get('status') == 'complete' and not alive(initial_state.get('pid')) and repair_process is None:
                if repair_state.get('status') != 'complete' and not alive(repair_state.get('pid')):
                    stream = (root / 'repair.log').open('a'); streams.append(stream)
                    env = worker_environment(resolve_device(gpu, inventory()), str(ROOT / 'src'))
                    repair_process = subprocess.Popen([sys.executable, __file__, '--output', str(root), '--phase', 'gpu', '--gpu', gpu],
                                                      cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                else:
                    repair_process = 'adopted'
            if repair_process is not None:
                if repair_process != 'adopted' and repair_process.poll() not in (None, 0):
                    resume_previous(root)
                    raise RuntimeError('Repair GPU worker failed; see repair.log')
                if repair_state.get('status') == 'complete' and not alive(repair_state.get('pid')):
                    resume_previous(root)
            summary = snapshot(root, 'running')
            if summary['counts'].get('repair_cases_complete') == len(plan['cases']):
                if alive(read(root / 'repair_worker.json', {}).get('pid')):
                    time.sleep(2)
                    continue
                resume_previous(root)
                final_summary = snapshot(root, 'complete')
                finalize(root)
                write_json(root / 'controller.json', {'status': final_summary['status'], 'pid': os.getpid()})
                break
            time.sleep(10)
    for stream in streams:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--phase', choices=('controller', 'initial', 'gpu', 'evaluate'), default='controller')
    parser.add_argument('--case-id')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase in ('initial', 'evaluate', 'gpu'):
        lock_directory = root / 'locks'
        lock_directory.mkdir(exist_ok=True)
        with (lock_directory / f'{args.phase}_{args.case_id or "worker"}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.phase == 'initial':
                evaluate_initial(root, args.case_id)
            elif args.phase == 'gpu':
                try:
                    gpu_repair(root, args.gpu)
                finally:
                    resume_previous(root)
            else:
                evaluate_repairs(root, args.case_id)
    else:
        with (root / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(root, args.gpu)
            except Exception as exc:
                write_json(root / 'controller.json', {'status': 'failed', 'pid': os.getpid(), 'error': f'{type(exc).__name__}: {exc}'})
                snapshot(root, 'failed')
                raise


if __name__ == '__main__':
    main()
