#!/usr/bin/env python3
"""Test-only deterministic evaluator. Native model does all neural inference in C++.
Tokenizer/score bookkeeping use SentencePiece/Python OUTSIDE the model runtime.
Optional externally supplied baseline JSON; never infer or fabricate GPT results.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import sentencepiece as sp

def normalize(text):
    text=text.replace("<end_of_turn>","").replace("<eos>","").strip()
    return re.sub(r"\s+"," ",text).strip().strip(" .!?,").casefold()

def evaluate_external(path, tasks):
    if path is None:
        return {"executed":False,"reason":"No independently collected GPT completions supplied"}
    base=json.loads(path.read_text())
    if not base.get("model") or not base.get("capture_method") or not base.get("timestamp"):
        raise ValueError("external baseline requires model, timestamp and capture_method")
    responses=base.get("responses")
    if not isinstance(responses,list):
        raise ValueError("external baseline requires a responses list")
    answers={}
    for v in responses:
        if v.get("id") in answers or not isinstance(v.get("answer"),str):
            raise ValueError("duplicate baseline response or non-string answer")
        answers[v["id"]]=v["answer"]
    expected={x["id"] for x in tasks}
    if set(answers)!=expected:
        raise ValueError("external baseline must include exactly all frozen task IDs")
    correct=sum(normalize(answers[t["id"]])==normalize(t["gold"]) for t in tasks)
    return {"executed":True,"model":base["model"],"capture_method":base["capture_method"],
            "timestamp":base["timestamp"],"passed":correct,"total":len(tasks),
            "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--sha",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-output-tokens",type=int,default=24)
    p.add_argument("--external-baseline-json",type=Path)
    a=p.parse_args()
    if a.out.exists():
        p.error("refusing to overwrite results")
    suite_bytes=a.suite.read_bytes()
    suite=json.loads(suite_bytes)
    tokenizer=sp.SentencePieceProcessor(model_file=str(a.tokenizer))
    rows=[]
    for i,item in enumerate(suite["tasks"]):
        text="<start_of_turn>user\n"+item["question"]+"<end_of_turn>\n<start_of_turn>model\n"
        ids=[2]+tokenizer.encode(text,out_type=int)
        if not ids or len(ids)>64:
            raise ValueError("prompt outside native max input: "+item["id"])
        command=[str(a.native),str(a.model),",".join(str(x) for x in ids),
                 str(a.max_output_tokens),"--expected-sha",a.sha]
        proc=subprocess.run(command,text=True,capture_output=True)
        if proc.returncode:
            row={**item,"passed":False,"error":proc.stderr[-2000:],"generated_ids":[]}
        else:
            ids_out=json.loads(proc.stdout)["generated_ids"]
            decoded=tokenizer.decode([t for t in ids_out if t not in (1,106)])
            row={**item,"prediction":decoded,"generated_ids":ids_out,
                 "input_ids":ids,"passed":normalize(decoded)==normalize(item["gold"])}
        rows.append(row)
        print(f'{i+1}/{len(suite["tasks"])} {item["id"]} {"PASS" if row["passed"] else "FAIL"}',flush=True)
    scores=defaultdict(lambda:{"correct":0,"total":0})
    for row in rows:
        v=scores[row["domain"]]
        v["correct"]+=int(row["passed"])
        v["total"]+=1
    summary={"scope":"Unseen-before-first-run local 28 question suite, now development data",
             "suite_sha256":hashlib.sha256(suite_bytes).hexdigest(),
             "source_model_sha256":a.sha,"native_model":"C++17 real Gemma3 270M (not joint toy planner)",
             "scoring":"strict normalized exact match; no LLM judge, no hints or post-hoc labels",
             "correct":sum(r["passed"] for r in rows),"total":len(rows),
             "by_domain":dict(scores),"gpt_baseline":evaluate_external(a.external_baseline_json,suite["tasks"]),
             "results":rows}
    a.out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2))

if __name__=="__main__":
    main()
