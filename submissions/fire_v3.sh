#!/usr/bin/env bash
# FIRE SCRIPT — v3-converged (the fleet's quad-node-verified candidate)
# Operator go received 2026-10-19: "submit it"
#
# Prereqs (on the firing node): kaggle CLI + a live KGAT/kaggle.json for the
# fleet account, quota OPEN (verified 2026-10-19: nothing fired since the
# banked 0.17 of Oct-18 19:14 UTC).
#
# Usage:  bash submissions/fire_v3.sh
set -euo pipefail

if ! command -v kaggle &> /dev/null; then
  echo "FATAL: kaggle CLI not found on PATH. Install it or configure credentials before firing." >&2
  exit 1
fi

REPO=Language-U-Gemma4
CAND=submissions/v3-converged/build/submission.zip
EXPECT_SHA=04cdcdfca4d23481618834a5f3facf4a134d023ec896f7aa8a4a5c40d4700e5e
DESC="GemmaAgent01-Fleet | solananft.art fleet v3-converged — V10 0.22-badge prompt + 4 evidenced splices + harness truncation fix 16384 | sha 04cdcdfc | quad-node verified (Zymatica/Julian/Amara/Kofi)"

echo "── 1. artifact byte-gate ──"
ACTUAL=$(sha256sum "$CAND" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECT_SHA" ]; then
  echo "FATAL: sha mismatch — expected $EXPECT_SHA, got $ACTUAL"; exit 1
fi
echo "sha gate PASS: 04cdcdfc…700e5e"

echo "── 2. quota gate ──"
kaggle competitions submissions -c gemma-4-developer-agent | head -5

echo "── 3. firing ──"
kaggle competitions submit -c gemma-4-developer-agent -f "$CAND" -m "$DESC"

echo "── 4. post-fire receipt ──"
# Sleep 10s to allow Kaggle backend ingestion/queue registration before querying submission receipt
sleep 10
kaggle competitions submissions -c gemma-4-developer-agent | head -6
echo "Fired. Post the new ref number to the fleet + Dan."
