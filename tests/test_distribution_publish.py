from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock


class DistributionPublishTests(TestCase):
    def test_failed_promotion_restores_previous_distribution(self):
        from scripts.publish_distribution import publish_distribution
        with TemporaryDirectory() as td:
            root = Path(td)
            stage, target = root / 'dist_pyinstaller.stage-test', root / 'dist_pyinstaller'
            stage.mkdir()
            target.mkdir()
            (stage / 'marker').write_text('new')
            (target / 'marker').write_text('previous')
            original = Path.rename
            def rename(source, destination):
                if source.resolve() == stage.resolve():
                    raise OSError('promotion failed')
                return original(source, destination)
            with mock.patch.object(Path, 'rename', rename):
                with self.assertRaises(OSError):
                    publish_distribution(root, stage, target)
            self.assertEqual((target / 'marker').read_text(), 'previous')
            self.assertEqual((stage / 'marker').read_text(), 'new')

    def test_success_retains_previous_distribution_as_backup(self):
        from scripts.publish_distribution import publish_distribution
        with TemporaryDirectory() as td:
            root = Path(td)
            stage, target = root / 'dist_pyinstaller.stage-test', root / 'dist_pyinstaller'
            stage.mkdir()
            target.mkdir()
            (stage / 'marker').write_text('new')
            (target / 'marker').write_text('previous')
            backup = publish_distribution(root, stage, target)
            self.assertEqual((target / 'marker').read_text(), 'new')
            self.assertEqual((backup / 'marker').read_text(), 'previous')

    def test_rejects_paths_outside_workspace(self):
        from scripts.publish_distribution import publish_distribution
        with TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                publish_distribution(root, root.parent / 'elsewhere', root / 'dist_pyinstaller')
