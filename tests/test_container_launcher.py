"""Offline contract checks for mount/device selection; no Docker invocation."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "docker/run_rk3588.py"
SPEC = importlib.util.spec_from_file_location("container_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ("assets with space", "work", "sdk/include", "sdk/lib"):
            (root / name).mkdir(parents=True)
        for name in ("sdk/include/rknn_api.h", "sdk/lib/librknnrt.so", "mali.so"):
            (root / name).touch()
        self.args = argparse.Namespace(assets=root / "assets with space", work=root / "work",
            rknn_sdk=root / "sdk", mali_lib=root / "mali.so", device=["/dev/mali0", "/dev/dri/renderD129"],
            image="experiment:pinned", interactive=False, dry_run=True,
            command=["--", "python3", "/work/source/tools/benchmark/run_micro.py", "--dry-run"])

    def test_mounts_and_payload_are_preserved(self):
        command = launcher.make_command(self.args)
        mounts = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--mount"]
        self.assertEqual(len(mounts), 4)
        self.assertEqual(sum(x.endswith(",readonly") for x in mounts), 3)
        self.assertIn("dst=/work", mounts[1])
        self.assertNotIn("--privileged", command)
        self.assertEqual(command[-4:], ["experiment:pinned", "python3", "/work/source/tools/benchmark/run_micro.py", "--dry-run"])

    def test_requires_explicit_devices(self):
        self.args.device = []
        with self.assertRaisesRegex(ValueError, "Select the actual"):
            launcher.make_command(self.args)

    def test_requires_explicit_command(self):
        self.args.command = []
        with self.assertRaisesRegex(ValueError, "explicit command"):
            launcher.make_command(self.args)

    def test_rejects_work_inside_assets(self):
        self.args.work = self.args.assets / "nested"
        self.args.work.mkdir()
        with self.assertRaisesRegex(ValueError, "non-nested"):
            launcher.make_command(self.args)

    def test_missing_driver_is_not_silently_substituted(self):
        self.args.mali_lib.unlink()
        with self.assertRaisesRegex(ValueError, "Missing dependency"):
            launcher.make_command(self.args)


if __name__ == "__main__":
    unittest.main()
