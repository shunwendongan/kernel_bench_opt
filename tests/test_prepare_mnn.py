"""Overlay-set resolution of scripts/prepare_mnn.py; no MNN checkout is touched."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_mnn.py"
SPEC = importlib.util.spec_from_file_location("prepare_mnn", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)

MANIFEST = {
    "schema_version": 2, "mnn_commit": "deadbeef", "patches": [],
    "common_overlays": [{"source": "src/x/common.cpp", "path": "engine/common.cpp", "before_sha256": None}],
    "default_overlay_set": "b",
    "overlay_sets": {
        "a": {"overlays": [{"source": "src/x/overlays/a/f.cpp", "path": "engine/f.cpp", "before_sha256": "0" * 64}]},
        "b": {"overlays": [{"source": "src/x/overlays/b/f.cpp", "path": "engine/f.cpp", "before_sha256": "0" * 64}]},
    },
}


class ResolveOverlays(unittest.TestCase):
    def test_default_set_prepends_common(self):
        name, overlays = PREPARE.resolve_overlays(MANIFEST)
        self.assertEqual(name, "b")
        self.assertEqual([o["source"] for o in overlays], ["src/x/common.cpp", "src/x/overlays/b/f.cpp"])

    def test_explicit_set(self):
        name, overlays = PREPARE.resolve_overlays(MANIFEST, "a")
        self.assertEqual(name, "a")
        self.assertEqual(overlays[-1]["source"], "src/x/overlays/a/f.cpp")

    def test_unknown_set_is_an_error(self):
        with self.assertRaises(RuntimeError):
            PREPARE.resolve_overlays(MANIFEST, "nope")

    def test_schema1_manifest_still_works(self):
        legacy = {"mnn_commit": "x", "patches": [], "overlays": [{"source": "s", "path": "p", "before_sha256": None}]}
        name, overlays = PREPARE.resolve_overlays(legacy)
        self.assertEqual((name, len(overlays)), ("default", 1))
        with self.assertRaises(RuntimeError):
            PREPARE.resolve_overlays(legacy, "a")

    def test_repository_manifest_resolves_every_set(self):
        if not (PREPARE.PATCHES / "manifest.json").is_file():
            self.skipTest("no engine overlay manifest on this branch (main)")
        manifest = PREPARE.load_manifest()
        for name in manifest.get("overlay_sets", {}):
            _, overlays = PREPARE.resolve_overlays(manifest, name)
            for item in overlays:
                self.assertTrue((PREPARE.REPO / item["source"]).is_file(), item["source"])


if __name__ == "__main__":
    unittest.main()
