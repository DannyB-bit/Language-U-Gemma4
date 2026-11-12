#!/usr/bin/env python3
"""35-Test Hardware Verification Suite for Trinity Hybrid Retrieval.

Executes physical test battery across:
  - Exact in-domain queries
  - Parent/hierarchical queries
  - Typos and morphological permutations
  - Adversarial out-of-domain noise (structural MISS)
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from .roadmap_hybrid_v5_gemma import HybridEngineV5, run_battery_in_process, s7_self_test
except ImportError:
    from roadmap_hybrid_v5_gemma import HybridEngineV5, run_battery_in_process, s7_self_test

def main():
    print("=" * 80)
    print("RUNNING 35-TEST PHYSICAL SILICON VERIFICATION SUITE")
    print("=" * 80)

    # 1. Self test
    print("\n[STEP 1] Physical S^7 Octonionic Compass Self-Test...")
    assert s7_self_test(), "S^7 self-test failed!"

    # 2. Battery test with physical Gemma-2 + Language-U S^7 compass
    no_gemma = os.environ.get("LUC_NO_GEMMA", "").strip() in ("1", "true", "yes")
    if no_gemma:
        print("\n[STEP 2] DEGRADED MODE (LUC_NO_GEMMA): initializing HybridEngineV5 without Gemma-2")
        print("         neural channel — 5-channel lane for torchless verification nodes.")
    else:
        print("\n[STEP 2] Initializing HybridEngineV5 with Google EmbeddingGemma-2 + Language-U S^7...")
    engine = HybridEngineV5(load_gemma=not no_gemma)

    print("\n[STEP 3] Running 35/35 Battery...")
    ok = run_battery_in_process(engine)
    if ok:
        print("\n[SUCCESS] 35/35 Battery PASSED with 0 False Positives!")
    else:
        print("\n[FAILURE] Battery test had failures!")
        sys.exit(1)

if __name__ == "__main__":
    main()
