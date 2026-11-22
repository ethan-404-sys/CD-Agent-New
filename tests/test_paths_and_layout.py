"""Verify unified CD-Agent single-package layout, path resolver, and clean imports."""
from pathlib import Path
import unittest

import cd_agent


class LayoutAndPathsTest(unittest.TestCase):
    def test_single_src_package(self):
        root = cd_agent.REPO_ROOT
        self.assertEqual(root.name, 'CD-Agent')
        src_dirs = sorted(p.name for p in (root / 'src').iterdir() if p.is_dir() and not p.name.startswith('__'))
        self.assertEqual(src_dirs, ['cd_agent'])

    def test_compat_does_not_reference_legacy_wyh_repo(self):
        compat = Path(cd_agent.__file__).resolve().parent / '_compat'
        self.assertTrue(compat.is_dir())
        self.assertNotIn('wyh/CD-Agent', str(compat))

    def test_model_and_dataset_symlinks_resolve(self):
        self.assertTrue(cd_agent.model_path('sam3').is_dir())
        self.assertTrue(cd_agent.model_path('SAM').is_dir())
        self.assertTrue(cd_agent.dataset_path('LEVIR-CD').is_dir())
        self.assertTrue(cd_agent.dataset_path('SECOND').is_dir())
        self.assertTrue(cd_agent.dataset_path('WHU-CD').is_dir())


if __name__ == '__main__':
    unittest.main()
