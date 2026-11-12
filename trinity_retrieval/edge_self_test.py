"""Honest edge self-test: import-gate battery for the Trinity engine family.

Verdict policy (THE REAL CODE LAW — no theater-green):
- every engine import is recorded PASS or FAIL with the real error
- exit code 0 ONLY if all engines import; otherwise exit 1 with an honest banner
- guarded-import engines report TORCH_AVAILABLE so a torchless node is a
  documented state, not a silent success
"""
import importlib
from pathlib import Path
import sys

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENGINES = [
    ("AirTrinityLayerStreamer", "air_trinity_streamer"),
    ("EdgeLoRAStreamer", "edge_lora_streamer"),
    ("SparseOctonionStreamer", "sparse_octonion_streamer"),
]

print("=== REMOTE PHYSICAL SILICON SELF-TEST ===")
print("Python executable:", sys.executable)

results = {}
for i, (label, module_name) in enumerate(ENGINES, start=1):
    try:
        mod = importlib.import_module(module_name)
        torch_state = getattr(mod, "TORCH_AVAILABLE", None)
        if torch_state is False:
            print(f"[{i}/{len(ENGINES)}] {label}: IMPORT OK, TORCH UNAVAILABLE "
                  "(guarded-import degrade — engine instantiation will raise)")
            results[label] = "degraded"
        else:
            print(f"[{i}/{len(ENGINES)}] {label}: OK")
            results[label] = "ok"
    except Exception as e:
        print(f"[{i}/{len(ENGINES)}] {label}: FAIL -", e)
        results[label] = "fail"

failed = [k for k, v in results.items() if v == "fail"]
degraded = [k for k, v in results.items() if v == "degraded"]
ok = [k for k, v in results.items() if v == "ok"]

print("=== VERIFICATION COMPLETE ===")
if failed:
    print(f"HONEST VERDICT: {len(failed)}/{len(ENGINES)} ENGINES FAILED TO IMPORT: {', '.join(failed)}")
    sys.exit(1)
if degraded:
    print(f"HONEST VERDICT: {len(ok)}/{len(ENGINES)} engines fully operational; "
          f"{len(degraded)} importable but torch-degraded on this node: {', '.join(degraded)}")
    print("This node cannot execute the degraded engines — run on a torch-equipped node for full execution.")
    sys.exit(1)
print(f"ALL {len(ENGINES)} ENGINES OPERATIONAL")
sys.exit(0)
