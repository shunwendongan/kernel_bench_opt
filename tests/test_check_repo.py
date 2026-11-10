"""Unit tests for the helpers of tools/verify/check_repo.py (no git or network)."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify/check_repo.py"
SPEC = importlib.util.spec_from_file_location("check_repo", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class Naming(unittest.TestCase):
    def test_candidate_ids(self):
        for good in ("bigvgan-deconv1d-v2", "predit-sync", "a1"):
            self.assertTrue(CHECK.CANDIDATE_ID_RE.match(good), good)
        for bad in ("BigVGAN-v2", "deconv_v2", "-x", "x-", "x--y"):
            self.assertFalse(CHECK.CANDIDATE_ID_RE.match(bad), bad)

    def test_run_ids(self):
        self.assertTrue(CHECK.RUN_ID_RE.match("20261110-150000-bigvgan-deconv1d-v2-l1"))
        self.assertTrue(CHECK.RUN_ID_RE.match("20261110-150000-omni-profile"))
        for bad in ("20261106-bigvgan-convtranspose", "20261110-150000-x-L1", "2026-11-10-x-l1"):
            self.assertFalse(CHECK.RUN_ID_RE.match(bad), bad)


class Validator(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["id", "status"],
              "properties": {"id": {"type": "string", "pattern": "^[a-z]+$"},
                             "status": {"type": "string", "enum": ["a", "b"]},
                             "n": {"type": ["integer", "null"]},
                             "items": {"type": "array", "items": {"type": "object", "required": ["p"]}}}}

    def test_valid(self):
        self.assertEqual(CHECK.validate({"id": "ok", "status": "a", "n": None, "items": [{"p": 1}]}, self.SCHEMA), [])

    def test_problems_are_reported_with_paths(self):
        problems = CHECK.validate({"id": "Bad1", "status": "c", "n": True, "items": [{}]}, self.SCHEMA)
        joined = "\n".join(problems)
        self.assertIn("$.id", joined)
        self.assertIn("$.status", joined)
        self.assertIn("$.n", joined)          # bool is not an integer
        self.assertIn("$.items[0]", joined)

    def test_repository_schemas_are_valid_json(self):
        for schema in (CHECK.SCHEMAS).glob("*.schema.json"):
            self.assertIsInstance(CHECK.schema(schema.name[:-len(".schema.json")]), dict)


if __name__ == "__main__":
    unittest.main()
