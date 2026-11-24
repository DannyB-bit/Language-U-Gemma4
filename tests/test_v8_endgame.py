import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "experiments" / "run270_rust" / "calib" / "31b"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V8EndgameTests(unittest.TestCase):
    def test_artifact_preflight_rejects_missing_and_corrupted_files(self):
        module = load_module("verify_v8_0_artifacts", CALIB / "verify_v8_0_artifacts.py")
        self.assertEqual(module._selftest(), 0)

    def test_evaluator_never_silently_drops_explicit_sidecars(self):
        module = load_module("eval_v6_1_skeptic_proof", CALIB / "eval_v6_1_skeptic_proof.py")
        self.assertEqual(module._run_selftest(), 0)


if __name__ == "__main__":
    unittest.main()
