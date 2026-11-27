"""Numerical fit and CLI geometry validation without launching Kit."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.contact_trials import fit_deceleration, parse_args


class DecelerationFitTests(unittest.TestCase):
    def test_recovers_effective_mu_and_residual(self):
        trace = [
            {'step': i + 1, 't_s': i * .01, 'p_m': [i * .001, 0, 0],
             'v_mps': [.6 - .25 * 9.81 * i * .01, 0, 0]}
            for i in range(25)
        ]
        result = fit_deceleration(trace)
        self.assertAlmostEqual(result['effective_dynamic_friction_fit'], .25, places=6)
        self.assertLess(result['fit_rmse_mps'], 1e-10)
        self.assertEqual(result['fit_samples'], 16)

    def test_insufficient_moving_samples_is_inconclusive(self):
        trace = [{'step': i+1, 't_s': i*.01, 'p_m': [0, 0, 0],
                  'v_mps': [0, 0, 0]} for i in range(12)]
        self.assertIsNone(fit_deceleration(trace)['effective_dynamic_friction_fit'])

    def test_cli_accepts_other_asset_and_custom_geometry(self):
        fake_app = types.ModuleType('isaaclab.app')
        fake_app.add_launcher_args = lambda parser: (parser.add_argument('--visualizer'), parser.add_argument('--device', default='cuda:0'))
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'isaaclab': types.ModuleType('isaaclab'), 'isaaclab.app': fake_app}):
            path = Path(root) / 'different.usd'
            path.touch()
            args = parse_args(['--trial', 'grasp', '--asset', str(path),
                               '--report', str(Path(root) / 'report.json'),
                               '--grasp-left-collider', 'Sides/Left',
                               '--grasp-right-collider', 'Sides/Right',
                               '--finger-open-x', '0.125',
                               '--finger-closed-x', '0.104'])
        self.assertEqual(args.finger_open_x, 0.125)
        self.assertEqual(args.grasp_left_collider, 'Sides/Left')
        self.assertEqual(args.device, 'cpu')
        self.assertFalse(args.device_explicit)

    def test_explicit_device_is_respected(self):
        fake_app = types.ModuleType('isaaclab.app')
        fake_app.add_launcher_args = lambda parser: parser.add_argument('--device', default='cuda:0')
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'isaaclab': types.ModuleType('isaaclab'), 'isaaclab.app': fake_app}):
            path = Path(root) / 'different.usd'
            path.touch()
            args = parse_args(['--trial', 'grasp', '--asset', str(path),
                               '--report', str(Path(root) / 'report.json'), '--device', 'cuda:0'])
        self.assertEqual(args.device, 'cuda:0')
        self.assertTrue(args.device_explicit)


if __name__ == '__main__':
    unittest.main()
