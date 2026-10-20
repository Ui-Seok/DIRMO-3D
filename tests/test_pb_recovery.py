"""Failure recovery must preserve unknown results and unrelated GPU jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dirmo.evaluation.pb_worker import bounded_measure
from dirmo.gpu_priority import set_saved_workers_paused

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_pb_guided_repair import seal_cpu_failure


class PBRecoveryTests(unittest.TestCase):
    def test_measurement_timeout_is_unknown_and_does_not_pass_or_invent_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdf, pocket = root / 'input.sdf', root / 'pocket.pdb'
            sdf.write_text('fixture'); pocket.write_text('fixture')
            with patch.dict(os.environ, {'DIRMO_PB_MEASURE_CACHE': str(root / 'cache'), 'DIRMO_PB_MEASUREMENT_TIMEOUT': '2'}), \
                 patch('dirmo.evaluation.pb_worker.subprocess.run', side_effect=subprocess.TimeoutExpired(['pb'], 2)) as run:
                result = bounded_measure(sdf, pocket, diagnose=True)
                self.assertFalse(result['passes'])
                self.assertTrue(result['missing'] and result['measurement_timeout'])
                self.assertEqual(result['checks'], {})
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                again = bounded_measure(sdf, pocket, diagnose=True)
                self.assertTrue(again['missing'])
                self.assertEqual(run.call_count, 1)

    def test_longer_measurement_budget_can_retry_a_previous_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdf, pocket = root / 'input.sdf', root / 'pocket.pdb'
            sdf.write_text('fixture'); pocket.write_text('fixture')
            with patch.dict(os.environ, {'DIRMO_PB_MEASURE_CACHE': str(root / 'cache'), 'DIRMO_PB_MEASUREMENT_TIMEOUT': '2'}), \
                 patch('dirmo.evaluation.pb_worker.subprocess.run', side_effect=subprocess.TimeoutExpired(['pb'], 2)):
                bounded_measure(sdf, pocket)

            def success(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                output.write_text(json.dumps({'passes': True, 'checks': {'example_check': True}}))
                return subprocess.CompletedProcess(command, 0)

            with patch.dict(os.environ, {'DIRMO_PB_MEASURE_CACHE': str(root / 'cache'), 'DIRMO_PB_MEASUREMENT_TIMEOUT': '3'}), \
                 patch('dirmo.evaluation.pb_worker.subprocess.run', side_effect=success) as run:
                result = bounded_measure(sdf, pocket)
                self.assertTrue(result['passes'])
                self.assertEqual(run.call_count, 1)

    def test_failed_cpu_case_preserves_completed_rows_and_keeps_unmeasured_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'study.json').write_text(json.dumps({'initial_draws': 3, 'repair_draws': 8}))
            result = root / 'evaluations' / 'initial' / 'case.json'
            result.parent.mkdir(parents=True)
            completed = {'draw': 0, 'generation_status': 'built', 'repair_eligible': False,
                         'pb': {'passes': True, 'checks': {'example_check': True}}}
            result.write_text(json.dumps({'case_id': 'case', 'draws': {'000': completed}}))
            seal_cpu_failure(root, 'initial', 'case', 'fixture timeout')
            saved = json.loads(result.read_text())
            self.assertEqual(saved['draws']['000'], completed)
            self.assertEqual(len(saved['draws']), 3)
            self.assertTrue(saved['evaluation_incomplete'])
            self.assertTrue(saved['draws']['001']['pb']['missing'])
            self.assertTrue((root / 'cases' / 'case' / 'requests_ready.json').exists())

    def make_pause(self, root):
        file = root / 'pause.json'
        record = {'workers': {str(pid): {'pid': pid, 'command': ['python', 'benchmark.py', '--gpu', '4']}
                              for pid in (123, 456)}}
        file.write_text(json.dumps(record))
        return file, record

    def test_reused_pid_prevents_every_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file, record = self.make_pause(root)
            commands = {123: record['workers']['123']['command'], 456: ['unrelated']}
            with patch('dirmo.gpu_priority.worker_command', side_effect=commands.__getitem__), \
                 patch('dirmo.gpu_priority.worker_state', return_value='R'), \
                 patch('dirmo.gpu_priority.os.kill') as kill:
                with self.assertRaises(RuntimeError):
                    set_saved_workers_paused(file, True, root, 'test')
                kill.assert_not_called()

    def test_pause_and_resume_signal_only_the_saved_gpu4_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file, record = self.make_pause(root)
            command = record['workers']['123']['command']
            with patch('dirmo.gpu_priority.worker_command', return_value=command), \
                 patch('dirmo.gpu_priority.worker_state', return_value='R'), \
                 patch('dirmo.gpu_priority.os.kill') as kill:
                changed = set_saved_workers_paused(file, True, root, 'repair ready')
                self.assertEqual(changed, [123, 456])
                self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGSTOP))
                self.assertEqual(kill.call_args_list[1].args, (456, signal.SIGSTOP))
            with patch('dirmo.gpu_priority.worker_command', return_value=command), \
                 patch('dirmo.gpu_priority.worker_state', return_value='T'), \
                 patch('dirmo.gpu_priority.os.kill') as kill:
                set_saved_workers_paused(file, False, root, 'CPU wait')
                self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGCONT))
                self.assertEqual(kill.call_args_list[1].args, (456, signal.SIGCONT))


if __name__ == '__main__':
    unittest.main()
