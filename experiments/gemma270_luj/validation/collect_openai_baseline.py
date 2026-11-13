#!/usr/bin/env python3
"""Collect an external GPT baseline JSON for run_unseen_eval.py (--external-baseline-json).

Fails closed when OPENAI_API_KEY is missing or the API returns an error; never fabricates answers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.server
import json
import os
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import urllib.error
import urllib.request


def build_template(suite_path: Path) -> dict:
    """Create an empty external-baseline template matching the suite's exact task IDs."""
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"invalid suite tasks in {suite_path}")
    return {
        "model": "<ENTER_NAMED_GPT_MODEL_E_G_gpt-4o>",
        "capture_method": "<ENTER_API_OR_UI_CAPTURE_DETAILS>",
        "timestamp": "<ISO-8601-UTC>",
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "responses": [{"id": str(item["id"]), "answer": ""} for item in tasks],
    }


def query_openai_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    """Call OpenAI-compatible chat/completions endpoint deterministically."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenAI response missing choices: {body}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"OpenAI response missing string content: {body}")
    return content.strip()


def collect_baseline(
    *,
    suite_path: Path,
    out_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    timeout_s: float,
    overwrite: bool,
) -> dict:
    """Query all suite prompts and write a validated external-baseline JSON."""
    if not api_key or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is unset or empty; refusing to fabricate GPT baseline")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {out_path}")

    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    tasks = suite["tasks"]
    responses = []
    for item in tasks:
        ans = query_openai_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            question=item["question"],
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        responses.append({"id": item["id"], "answer": ans})

    record = {
        "model": model,
        "capture_method": f"openai_chat_completions_temperature_0@{base_url.rstrip('/')}",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "responses": responses,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def run_self_test() -> None:
    """Exercise template generation, missing-key fail-closed behavior, and local mock HTTP collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        suite_path = tmp / "mini_suite.json"
        suite_path.write_text(
            json.dumps(
                {
                    "name": "mini",
                    "tasks": [
                        {"id": "q1", "domain": "arithmetic", "question": "2+2?", "gold": "4"},
                        {"id": "q2", "domain": "factual", "question": "Capital of Japan?", "gold": "Tokyo"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        tmpl = build_template(suite_path)
        assert len(tmpl["responses"]) == 2
        assert tmpl["responses"][0] == {"id": "q1", "answer": ""}

        out_path = tmp / "baseline.json"
        failed_missing_key = False
        try:
            collect_baseline(
                suite_path=suite_path,
                out_path=out_path,
                model="gpt-test",
                api_key="",
                base_url="http://127.0.0.1:1/v1",
                max_tokens=32,
                timeout_s=2.0,
                overwrite=False,
            )
        except ValueError:
            failed_missing_key = True
        assert failed_missing_key, "empty API key must fail closed"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                req_json = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = req_json["messages"][0]["content"]
                reply = "4" if "2+2" in prompt else "Tokyo"
                body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as srv:
            port = srv.server_address[1]
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                rec = collect_baseline(
                    suite_path=suite_path,
                    out_path=out_path,
                    model="gpt-test-local",
                    api_key="test-key-local",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    max_tokens=16,
                    timeout_s=5.0,
                    overwrite=False,
                )
            finally:
                srv.shutdown()

        assert rec["model"] == "gpt-test-local"
        assert [r["id"] for r in rec["responses"]] == ["q1", "q2"]
        assert [r["answer"] for r in rec["responses"]] == ["4", "Tokyo"]
    print("COLLECT_OPENAI_BASELINE_SELFTEST=PASS")


def main() -> int:
    p = argparse.ArgumentParser(description="Collect external OpenAI GPT baseline JSON (fail-closed).")
    p.add_argument("--suite", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--write-template", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not args.suite or not args.out:
        p.error("--suite and --out are required unless --self-test is given")

    if args.write_template:
        if args.out.exists() and not args.overwrite:
            p.error(f"refusing to overwrite existing template: {args.out}")
        args.out.write_text(json.dumps(build_template(args.suite), indent=2) + "\n", encoding="utf-8")
        return 0

    try:
        collect_baseline(
            suite_path=args.suite,
            out_path=args.out,
            model=args.model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
