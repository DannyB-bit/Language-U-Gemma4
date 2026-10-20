#!/usr/bin/env python3
"""Fleet wheelhouse A/B evaluator — Kaggle 4x L4 kernel (NOT a leaderboard submission).

Purpose: measure candidate submission bundles against a control on the SAME
paired task set, with the OFFICIAL stack — the wheelhouse's vLLM 0.19.1
(TP=4, flags from the organizers' Getting Started notebook) + swegemma
Evaluator with the subprocess sandbox, each submission under its own
eval_config.yaml, exactly like the scorer. API surface adapted from
happyc0der's proven kaggle/eval_official_l4/run.py (public, Apache-2.0).

Outputs: /kaggle/working/run.log, /kaggle/working/results/<name>/
(task_results.jsonl, summary.json, logs/), paired_ab.json, <name>.tgz
"""
import asyncio, glob, json, os, pathlib, random, subprocess, sys, time

T0 = time.time()

# ── THE MATRIX (edit here; each entry = one eval lane) ────────────────────────
# Control first. Every lane runs the SAME paired task subset (TASK_SEED below).
RUNS = [
    ("ctrl-v5", "submissions/v5-restored"),           # control: TRUE 0.17-lineage prompt + zero-coupling fixes (sha 0b32f172…)
    ("cand-v5-arch", "submissions/v5-analyzer-coder"), # candidate: analyzer+coder two-agent, 0.24-lineage (sha 60b98087…) — PR #15
    ("cand-v4", "submissions/v4-loop-guard"),         # candidate: v3/v11-family prompt + loop-guards (sha 7e89e9c0…)
    # ("cand-v2", "submissions/v2-measured"),         # optional: the ablation budget climb
    # ("cand-unban", "submissions/v6-graphunban"),    # optional: the graph-tool lever (to build)
]
REPO_URL = "https://github.com/DannyB-bit/Language-U-Gemma4.git"
TASK_COUNT = 40        # paired-subset size (budget: ~2.5-4 h per lane)
TASK_SEED = 20261019   # fixed seed → reproducible subset across lanes + reruns
SESSION_BUDGET_H = 11.0
EST_RUN_H = 4.0        # conservative per-lane estimate at 8 min/task ceiling

LOG = open("/kaggle/working/run.log", "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n")
def sh(cmd, timeout=None, check=False):
    log("$", cmd[:240])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout[-2000:] + ("\n" + r.stderr[-2000:] if r.stderr.strip() else "")).strip()
    if out: log(out)
    if check and r.returncode != 0: raise SystemExit(f"command failed rc={r.returncode}: {cmd[:120]}")
    return r
def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True)); return hits[0] if hits else None

# 0. hardware: ADAPT to what Kaggle provisioned (AMARA's PR #15 fix — the 2026-10-19
#    run hard-aborted here on 2x T4 and the fleet fired v3 without an A/B verdict;
#    paired same-hardware verdicts stay decision-valid, only absolute rates read low).
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2")
ngpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
gpu_names = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                            capture_output=True, text=True).stdout
log("GPUs:", ngpu, "|", gpu_names.strip().replace("\n", "; "))
if ngpu < 1:
    log("FATAL: no GPUs at all. Stopping."); sys.exit(0)
TP = ngpu                                  # tensor parallel = what we actually have (was: hardcoded 4)
IS_T4 = "T4" in gpu_names
IS_SCORER_SHAPE = (ngpu >= 4 and not IS_T4)
if not IS_SCORER_SHAPE:
    TASK_COUNT = min(TASK_COUNT, 20)
    EST_RUN_H = max(EST_RUN_H, 5.0)
    log(f"note: non-scorer shape ({ngpu}x {'T4' if IS_T4 else 'GPU'}); "
        f"tp={TP}, paired subset scaled to {TASK_COUNT} tasks, per-lane estimate {EST_RUN_H} h. "
        f"Verdicts from this shape are PAIRED (same hardware both lanes) so the A/B delta stays valid; "
        f"absolute pass-rates will read lower than the scorer's. For scorer-faithful numbers request "
        f"'GPU L4 x4' in the Kaggle notebook Settings > Accelerator before pushing this kernel.")

# 1. inputs
tasks = first("/kaggle/input/**/tasks.jsonl"); assert tasks, "competition data not found"
DATA = pathlib.Path(tasks).parent; log("DATA =", DATA)
whl = first("/kaggle/input/**/swegemma-*.whl"); assert whl, "wheelhouse not found"
WHEELHOUSE = pathlib.Path(whl).parent; log("WHEELHOUSE =", WHEELHOUSE)
cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
       if "31b" in p.lower() and "w4a16" in p.lower()]
assert cfg, "31B W4A16 model not attached"; MODEL_PATH = str(pathlib.Path(sorted(cfg)[0]).parent)
log("MODEL_PATH =", MODEL_PATH)

# 2. environment, as in the Getting Started notebook; torch pins vLLM 0.19.1
for k, v in {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
             "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
             "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_NO_USAGE_STATS": "1",
             "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
    os.environ[k] = v
tv = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                    capture_output=True, text=True).stdout.strip()
log("preinstalled torch:", tv)
if not tv.startswith("2.10.0"):
    sh(f"{sys.executable} -m pip install -q torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
       f"--index-url https://download.pytorch.org/whl/cu128", timeout=3600, check=True)
for pth in glob.glob("/usr/local/lib/python*/dist-packages/*cutlass*.pth") + \
           glob.glob("/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    try: os.unlink(pth)
    except OSError: pass
tmp = pathlib.Path("/tmp/wheelhouse"); tmp.mkdir(exist_ok=True)
for w in WHEELHOUSE.glob("*.whl"):
    if "cutlass" in w.name.lower(): continue
    name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    if not (tmp / name).exists(): os.symlink(w, tmp / name)
sh(f"{sys.executable} -m pip install -q --no-deps --force-reinstall "
   f"{' '.join(sorted(str(w) for w in tmp.glob('*.whl')))}", timeout=3600, check=True)
sh(f"{sys.executable} -c 'import torch, vllm, swegemma, google.adk; "
   f"print(\"torch\", torch.__version__, \"vllm\", vllm.__version__, \"adk\", google.adk.__version__)'", check=True)

# 3. candidates: private fleet repo → token via Kaggle User Secret 'github-token'
#    (fallback: GITHUB_TOKEN env, or a pre-attached /kaggle/input dataset copy)
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("github-token") or None
except Exception:
    tok = os.environ.get("GITHUB_TOKEN") or None
work = pathlib.Path("/kaggle/working")
repo = work / "Language-U-Gemma4"
if not repo.exists():
    if tok:
        sh(f"cd {work} && git clone -q https://x-access-token:{tok}@{REPO_URL.split('https://')[1]} "
           f"&& cd Language-U-Gemma4 && git log --oneline | head -1", check=True)
    else:
        # dir-mode zip datasets arrive as .zip files — extract them first
        import zipfile as _zf
        _cands = pathlib.Path("/kaggle/working/cands"); _cands.mkdir(parents=True, exist_ok=True)
        for _zp in glob.glob("/kaggle/input/language-u-gemma4-candidates/*.zip"):
            with _zf.ZipFile(_zp) as _z:
                _z.extractall(_cands)
            log("extracted", _zp, "→", _cands)
        alt = first("/kaggle/input/**/v5-restored/eval_config.yaml") or \
              first("/kaggle/working/cands/**/v5-restored/eval_config.yaml")
        assert alt, "no github-token secret and no attached candidates dataset"
        repo = pathlib.Path(alt)
        while repo.parent != repo and not (repo / "submissions" / "v5-restored").is_dir():
            repo = repo.parent
        assert (repo / "submissions" / "v5-restored").is_dir(), "candidates root not found"
        log("using attached candidates at", repo)

# 4. the paired task subset: seeded, sorted, reproducible
all_ids = [json.loads(l)["instance_id"] for l in open(tasks) if l.strip()]
sub_ids = sorted(random.Random(TASK_SEED).sample(sorted(all_ids), min(TASK_COUNT, len(all_ids))))
log(f"paired subset: {len(sub_ids)} of {len(all_ids)} tasks (seed {TASK_SEED})")

# 5. vLLM server, flags from the organizers' Getting Started notebook
import litellm, torch, yaml
litellm.drop_params = True
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models.discovery import validate_single_declared_model
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
TARGET = "gemma-4-31b-it-qat-w4a16-ct"
first_sub = repo / RUNS[0][1]
server = VllmServer(VllmConfig(
    model=MODEL_PATH, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    default_chat_template_kwargs={"enable_thinking": True}, max_model_len=32768,
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto", gpu_memory_utilization=0.90,
    enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=TP, startup_timeout=60 * 20),
    adapter_manifest=discover_adapters(str(first_sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS))
ts = time.time(); server.start(); log(f"vLLM up on {server.base_url} (tp={TP}) after {time.time()-ts:.0f}s")

# 6. evaluate each lane on the SAME subset, sequentially, own eval_config.yaml
def evaluate(name, rel):
    sub = repo / rel
    ev = yaml.safe_load((sub / "eval_config.yaml").read_text()); ev = ev.get("evaluation", ev)
    declared = validate_single_declared_model(sub)
    adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    models = server.create_model_registry(aliases=[declared, TARGET], model_prefix="openai/", api_key="EMPTY")
    limits, gen = build_submission_limits()
    turns = ev.get("max_turns", ev.get("max_llm_calls"))
    cfg = EvalConfig(
        tasks_path=pathlib.Path(tasks), snapshots_dir=DATA / "snapshots",
        results_dir=pathlib.Path(f"/kaggle/working/results/{name}"),
        submission_dir=sub, models=models, sandbox="subprocess",
        timeout_seconds=ev.get("timeout_seconds"), max_time_minutes=ev.get("max_time_minutes"),
        max_tool_calls=ev.get("max_tool_calls"), max_turns=int(turns) if turns is not None else None,
        limits=limits, generation_constraints=gen, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                         token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"), wheels_dir=DATA / "wheels",
        task_ids=sub_ids, concurrency=1, display_mode="quiet", verbose=False)
    log(f"RUN {name}: {rel} eval={ev} tasks={len(sub_ids)}")
    t = time.time(); res = asyncio.run(Evaluator(cfg).run())
    log(f"RUN_DONE {name}: resolved {res.resolved}/{res.total} in {(time.time()-t)/60:.0f} min")
    sh(f"cd /kaggle/working/results && tar -czf /kaggle/working/{name}.tgz {name}")
    return res

done = {}
for name, rel in RUNS:
    elapsed_h = (time.time() - T0) / 3600
    if elapsed_h + EST_RUN_H > SESSION_BUDGET_H:
        log(f"SKIP {name}: {elapsed_h:.1f} h used, not enough session left"); continue
    try:
        done[name] = evaluate(name, rel)
    except Exception as e:
        log(f"RUN_FAILED {name}: {type(e).__name__}: {e}")
server.stop() if hasattr(server, "stop") else None

# 7. paired A/B read-out + pre-registered decision rule
def per_task(name):
    p = pathlib.Path(f"/kaggle/working/results/{name}/task_results.jsonl")
    if not p.exists(): return {}
    return {json.loads(l)["instance_id"]: json.loads(l) for l in open(p) if l.strip()}

if len(done) >= 2:
    ctrl = RUNS[0][0]
    table = {}
    for name in done:
        table[name] = {tid: bool(r.get("resolved")) for tid, r in per_task(name).items()}
    pairs = [(tid, {n: table[n].get(tid) for n in table}) for tid in sub_ids]
    out = {"seed": TASK_SEED, "n_tasks": len(sub_ids), "lanes": {n: int(sum(v.values())) for n, v in table.items()},
           "paired": [{"task": t, "lanes": d} for t, d in pairs]}
    for name in list(done)[1:]:
        delta = sum(1 for t, d in pairs if d.get(name) and not d.get(ctrl))
        loss   = sum(1 for t, d in pairs if d.get(ctrl) and not d.get(name))
        ties   = len(pairs) - delta - loss
        verdict = "SHIP (>= +3 outside noise)" if delta - loss >= 3 else \
                  ("INCONCLUSIVE (keep incumbent)" if delta - loss >= -2 else "DISCARD")
        out[f"ab_{name}_vs_{ctrl}"] = {"gained": delta, "lost": loss, "net": delta - loss, "ties": ties,
                                       "decision_rule": "ship only if net >= +3 on the same paired set",
                                       "verdict": verdict}
        log(f"PAIRED {name} vs {ctrl}: +{delta} / -{loss} (net {delta-loss:+d}) → {verdict}")
    json.dump(out, open("/kaggle/working/paired_ab.json", "w"), indent=1)
    log("paired_ab.json written")
log(f"ALL_DONE after {(time.time()-T0)/3600:.2f} h")
