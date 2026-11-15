#!/usr/bin/env python3
"""Parity tests linking Python production modules to Lean 4 formal contracts in TrinityGates.lean."""

import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trinity_retrieval import roadmap_hybrid_v5_gemma as v5
from trinity_retrieval.noise_stability_sieve import NoiseStabilitySieve
from neon_runner.container_sniff import MODE_TO_T16C, tile16_blk_bytes

LEAN_DIR = REPO_ROOT / "trinity_retrieval" / "formal_verification"
LEAN_FILE = LEAN_DIR / "TrinityGates.lean"
BP4 = 10_000


def _lean_def_int(src: str, name: str) -> int:
    m = re.search(rf"(?:def\s+{re.escape(name)}(?:\s*:\s*Nat)?|{re.escape(name)})\s*:=\s*(\d+)", src)
    if not m:
        raise AssertionError(f"Constant {name!r} not found in {LEAN_FILE}")
    return int(m.group(1))


def _py_tier_from_lean_semantics(score_bp: int, lex_bp: int, is_noise: bool, sieve_verdict: str, caller_opt_in: bool):
    if score_bp >= int(round(v5.STRONG_GATE * BP4)) and lex_bp >= int(round(v5.LEX_STRONG_FLOOR * BP4)):
        base = "STRONG"
    elif score_bp >= int(round(v5.WEAK_GATE * BP4)) and lex_bp >= int(round(v5.LEX_WEAK_FLOOR * BP4)):
        base = "WEAK"
    else:
        base = "MISS"

    if is_noise or sieve_verdict == "REJECT_MISS":
        eff = "MISS"
    elif sieve_verdict == "FORCE_WEAK" and base == "STRONG":
        eff = "WEAK"
    else:
        eff = base

    admitted = (eff == "STRONG") or (eff == "WEAK" and caller_opt_in)
    return eff, admitted


class TestLeanSpecParity(unittest.TestCase):
    def test_no_sorry_or_custom_axioms_in_lean_spec(self):
        src = LEAN_FILE.read_text(encoding="utf-8")
        # Strip line comments and block comments so doc prose is ignored.
        no_block = re.sub(r"/-.*?-/", "", src, flags=re.S)
        code_lines = [line.split("--", 1)[0] for line in no_block.splitlines()]
        code_only = "\n".join(code_lines)
        self.assertNotRegex(code_only, r"\bsorry\b", "Lean formalization must contain zero `sorry` placeholders")
        self.assertNotRegex(code_only, r"^\s*axiom\b", "Lean formalization must declare zero custom axioms")
        self.assertNotRegex(code_only, r"\bnative_decide\b", "Use kernel-checked `decide`/`rfl`/`omega`, not `native_decide`")

    def test_python_and_lean_constants_match_exactly(self):
        src = LEAN_FILE.read_text(encoding="utf-8")
        expected = {
            "w_bm25": int(round(v5.W_BM25 * BP4)),
            "w_sym": int(round(v5.W_SYM * BP4)),
            "w_char": int(round(v5.W_CHAR * BP4)),
            "w_lsa": int(round(v5.W_LSA * BP4)),
            "w_gemma": int(round(v5.W_GEMMA * BP4)),
            "w_s7": int(round(v5.W_S7 * BP4)),
            "tau_strong_score": int(round(v5.STRONG_GATE * BP4)),
            "tau_weak_score": int(round(v5.WEAK_GATE * BP4)),
            "lex_strong_floor": int(round(v5.LEX_STRONG_FLOOR * BP4)),
            "lex_weak_floor": int(round(v5.LEX_WEAK_FLOOR * BP4)),
        }
        for field_name, py_val in expected.items():
            lean_val = _lean_def_int(src, field_name)
            self.assertEqual(lean_val, py_val, f"Mismatch for {field_name}: Lean={lean_val} vs Python={py_val}")

        sieve = NoiseStabilitySieve(vocab={"alpha", "beta"})
        self.assertEqual(_lean_def_int(src, "SIEVE_PASS_TH"), int(round(sieve.pass_threshold * BP4)))
        self.assertEqual(_lean_def_int(src, "SIEVE_WEAK_TH"), int(round(sieve.weak_threshold * BP4)))

    def test_tile16_container_byte_theorems_match_python(self):
        src = LEAN_FILE.read_text(encoding="utf-8")
        theorem_map = {
            0: "tile16_mode0_1b_asym_bytes",
            1: "tile16_mode1_tern5_bytes",
            2: "tile16_mode2_2b_sym_bytes",
            3: "tile16_mode3_2b_asym_bytes",
            4: "tile16_mode4_3b_sym_bytes",
        }
        for mode_i, (n_bits, is_asym) in MODE_TO_T16C.items():
            py_bytes = tile16_blk_bytes(256, n_bits, is_asym)
            m = re.search(rf"theorem\s+{theorem_map[mode_i]}\s*:\s*tile16BlkBytes\s+256\s+{n_bits}\s+{'true' if is_asym else 'false'}\s*=\s*(\d+)", src)
            self.assertIsNotNone(m, f"Missing Lean theorem for mode {mode_i}")
            self.assertEqual(int(m.group(1)), py_bytes)

    def test_randomized_gate_invariants_match_lean_theorems(self):
        rng = random.Random(20261114)
        for _ in range(1000):
            score_bp = rng.randint(0, 10000)
            lex_bp = rng.randint(0, 10000)
            is_noise = rng.choice([False, False, True])
            verdict = rng.choice(["PASS", "FORCE_WEAK", "REJECT_MISS"])
            opt_in = rng.choice([False, True])
            eff, admitted = _py_tier_from_lean_semantics(score_bp, lex_bp, is_noise, verdict, opt_in)

            if is_noise or verdict == "REJECT_MISS" or lex_bp < 500 or score_bp < 2800:
                self.assertEqual(eff, "MISS")
                self.assertFalse(admitted)
            if lex_bp < 800 or verdict == "FORCE_WEAK":
                self.assertNotEqual(eff, "STRONG")
            if admitted and eff == "WEAK":
                self.assertTrue(opt_in)
            if not opt_in and admitted:
                self.assertEqual(eff, "STRONG")

    def test_lean_or_lake_verifies_trinity_gates_when_installed(self):
        elan_bin = Path.home() / ".elan" / "bin"
        env = os.environ.copy()
        if elan_bin.is_dir():
            env["PATH"] = str(elan_bin) + os.pathsep + env.get("PATH", "")
        lake_exe = shutil.which("lake", path=env.get("PATH"))
        lean_exe = shutil.which("lean", path=env.get("PATH"))
        if lake_exe is None and lean_exe is None:
            self.skipTest("Neither `lake` nor `lean` binary is installed in this environment")
        theorems = [
            "Trinity.canonConfig_wellFormed",
            "Trinity.sieve_reject_when_s1_too_low",
            "Trinity.noise_refusal_guarantee",
            "Trinity.sieve_reject_forces_miss",
            "Trinity.zero_confabulation_guarantee",
            "Trinity.weak_lexical_floor_guarantee",
            "Trinity.pure_neural_zero_lex_never_admitted",
            "Trinity.weak_requires_opt_in",
            "Trinity.opt_out_never_emits_weak",
            "Trinity.sieve_force_weak_prevents_strong",
            "Trinity.score_below_weak_always_miss",
            "Trinity.strong_preserved_under_increase",
            "Trinity.admitted_monotone_pass",
            "Trinity.tile16_mode1_tern5_bytes",
            "Trinity.pi4_v3_ram_budget_fits_cold_boot",
        ]
        if lake_exe is not None:
            res = subprocess.run([lake_exe, "build"], cwd=str(LEAN_DIR), env=env, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, f"lake build failed:\n{res.stdout}\n{res.stderr}")
            axiom_probe = "import TrinityGates\n" + "\n".join(f"#print axioms {t}" for t in theorems) + "\n"
            ax_res = subprocess.run(
                [lake_exe, "env", "lean", "--stdin"],
                cwd=str(LEAN_DIR),
                env=env,
                input=axiom_probe.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            ax_out = ax_res.stdout.decode("utf-8", errors="replace")
            ax_err = ax_res.stderr.decode("utf-8", errors="replace")
            self.assertEqual(ax_res.returncode, 0, f"lake env lean --stdin failed:\n{ax_out}\n{ax_err}")
        else:
            src_plus_ax = LEAN_FILE.read_text(encoding="utf-8") + "\n" + "\n".join(f"#print axioms {t}" for t in theorems) + "\n"
            ax_res = subprocess.run(
                [lean_exe, "--stdin"],
                cwd=str(LEAN_DIR),
                env=env,
                input=src_plus_ax.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            ax_out = ax_res.stdout.decode("utf-8", errors="replace")
            ax_err = ax_res.stderr.decode("utf-8", errors="replace")
            self.assertEqual(ax_res.returncode, 0, f"lean --stdin check failed:\n{ax_out}\n{ax_err}")

        combined = ax_out + "\n" + ax_err
        for forbidden in ("sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"):
            self.assertNotIn(forbidden, combined, f"Forbidden axiom {forbidden!r} detected:\n{combined}")
        allowed_axioms = {"propext", "Classical.choice", "Quot.sound"}
        for thm in theorems:
            m = re.search(rf"'{re.escape(thm)}'\s+(does not depend on any axioms|depends on axioms:\s*\[([^\]]+)\])", ax_out)
            self.assertIsNotNone(m, f"Missing #print axioms output for {thm}:\n{ax_out}")
            if m.group(2):
                used = {tok.strip() for tok in m.group(2).split(",") if tok.strip()}
                self.assertTrue(used.issubset(allowed_axioms), f"Unexpected axiom(s) in {thm}: {used - allowed_axioms}")


if __name__ == "__main__":
    unittest.main()
