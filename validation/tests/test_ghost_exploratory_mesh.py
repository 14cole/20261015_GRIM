"""Single certification candidates must remain usable without certification."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'GHOST'))
from ghost_backend import run_local_monostatic_exploratory as runner
from ghost_backend.execution.options import validate_for_run
from ghost_backend.runs.exploratory import select_exploratory_mesh
from ghost_backend.runs.inputs import load_geometry_snapshot
from ghost_backend.twod import solver
from ghost_backend.twod.adaptive_geometry import candidate_meshes
from ghost_backend.twod.preparation import prepare_geometry

FIXTURE = ROOT / 'tools/GHOST/ghost_backend/validation/material_examples/type2_pec.geo'


def geometry(large=False):
    snapshot, base = load_geometry_snapshot(str(FIXTURE), {})
    if large:
        for segment in snapshot['segments']:
            for pair in segment['point_pairs']:
                for key in ('x1', 'y1', 'x2', 'y2'):
                    pair[key] *= 20
    return snapshot, base


class ExploratoryMeshTests(unittest.TestCase):
    def test_initial_candidates_match_certification_geometry_and_degree(self):
        source, base = geometry(large=True)
        original = deepcopy(source)
        _, _, materials, scale = prepare_geometry(source, base, 'meters')
        expected = candidate_meshes(source, materials, 1.5, True, [1.], scale)
        for index, mode in enumerate(('certification_base', 'certification_fine')):
            with self.subTest(mode=mode):
                mesh, options, record = select_exploratory_mesh(
                    source, 1., mesh_mode=mode, geometry_units='meters', material_base_dir=base)
                self.assertEqual(mesh, expected[index][1])
                self.assertEqual(options['basis_order'], expected[index][2])
                self.assertEqual(options['mesh_strategy'], 'local')
                self.assertEqual(options['factorization'], 'adaptive')
                self.assertEqual(options['compressed_storage_mib'], 0)
                self.assertFalse(record['certification_performed'])
                validate_for_run(options, method='auto', precision='double')
                mesh['segments'][0]['properties'][1] = '-99'
                self.assertEqual(source, original)

    def test_small_geometry_uses_linear_candidate_and_explains_why(self):
        source, base = geometry()
        for mode, density in (('certification_base', '0'), ('certification_fine', '-30')):
            with self.subTest(mode=mode):
                mesh, options, record = select_exploratory_mesh(source, 1., mesh_mode=mode,
                    geometry_units='meters', material_base_dir=base)
                self.assertEqual(options['basis_order'], 1)
                self.assertEqual(options['mesh_strategy'], 'global')
                self.assertEqual(str(mesh['segments'][0]['properties'][1]), density)
                self.assertIn('Small reference mesh', record['selection_reason'])
                self.assertFalse(record['higher_order_candidate'])

    def test_mesh_selection_is_frequency_local(self):
        source, base = geometry(large=True)
        degrees = []
        for frequency in (.1, 1.):
            _, options, _ = select_exploratory_mesh(source, frequency,
                geometry_units='meters', material_base_dir=base)
            degrees.append(options['basis_order'])
        self.assertEqual(degrees, [1, 2])

    def test_explicit_panel_counts_are_preserved_in_base_candidate(self):
        source, base = geometry(large=True)
        source['segments'][0]['properties'][1] = '16'
        mesh, options, record = select_exploratory_mesh(source, 1.,
            geometry_units='meters', material_base_dir=base)
        self.assertEqual(mesh, source)
        self.assertEqual(options['basis_order'], 1)
        self.assertIn('Explicit fixed panel counts', record['selection_reason'])

    def test_geometry_mode_preserves_original_mesh_and_supports_old_dense_choice(self):
        source, base = geometry(large=True)
        mesh, options, record = select_exploratory_mesh(source, 1., mesh_mode='geometry',
            geometry_units='meters', material_base_dir=base, factorization='dense', ram_budget_gib=2.)
        self.assertEqual(mesh, source)
        self.assertIsNot(mesh, source)
        self.assertEqual(options['factorization'], 'dense')
        self.assertEqual(options['basis_order'], 1)
        self.assertEqual(options['ram_budget_gib'], 2.)
        self.assertEqual(record['candidate'], 'geometry')

    def test_invalid_mode_or_resource_options_are_not_silently_ignored(self):
        source, base = geometry()
        for options in ({'mesh_mode': 'typo'}, {'ram_budget_gib': -1}, {'factorization': 'typo'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_exploratory_mesh(source, 1., **options)
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            select_exploratory_mesh(source, float('nan'))

    def test_each_initial_candidate_runs_and_exports_without_certification(self):
        # The default square scaled to 2 m has >512 P1 reference panels, so it
        # takes the real coarsened quadratic/cubic path rather than a fallback.
        text = FIXTURE.read_text().replace('-0.05', '-1.0').replace('0.05', '1.0')
        for mode, degree in (('certification_base', 2), ('certification_fine', 3)):
            with self.subTest(mode=mode), TemporaryDirectory() as temp:
                geo_path = Path(temp) / 'square.geo'
                geo_path.write_text(text)
                output = io.StringIO()
                with patch.multiple(runner, GEOMETRY_FILE=str(geo_path),
                        FREQUENCIES_GHZ=[1.], AZIMUTHS_DEG=[0., 45.], GEOMETRY_UNITS='meters',
                        OUTPUT_DIR=str(Path(temp) / 'results'), MESH_MODE=mode), \
                     patch.object(solver, 'solve_monostatic_rcs_2d_certified',
                                  side_effect=AssertionError('Certification must not run')), \
                     patch.object(solver, 'solve_monostatic_rcs_2d',
                                  wraps=solver.solve_monostatic_rcs_2d) as solve, \
                     redirect_stdout(output):
                    run_dir = runner.main()
                self.assertEqual(solve.call_count, 1)
                kwargs = solve.call_args.kwargs
                self.assertFalse(kwargs['strict_quality_gate'])
                self.assertFalse(kwargs['compute_condition_number'])
                self.assertEqual(kwargs['solver_method'], 'auto')
                self.assertEqual(kwargs['execution_options']['basis_order'], degree)
                paths = list(run_dir.glob('*.grim'))
                self.assertTrue(paths)
                for path in paths:
                    with np.load(path, allow_pickle=False) as saved:
                        metadata = json.loads(str(saved['solver_metadata_json'].item()))['metadata']
                        self.assertFalse(metadata['mesh_convergence_certified'])
                        self.assertEqual(metadata['polynomial_degree'], degree)
                        self.assertLess(metadata['panel_count'], 100)
                        self.assertEqual(metadata['exploratory_mesh']['requested'], mode)
                        self.assertFalse(metadata['exploratory_mesh']['certification_performed'])
                        self.assertIn('no mesh certification', str(saved['history'].item()))
                self.assertIn('No convergence comparison or certification', output.getvalue())
                self.assertEqual(geo_path.read_text(), text)


if __name__ == '__main__':
    unittest.main()
