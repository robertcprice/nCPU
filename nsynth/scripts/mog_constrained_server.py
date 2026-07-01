#!/usr/bin/env python3
"""Drop-in OpenAI-compatible server that decodes GRAMMAR-CONSTRAINED Mog.

Exposes the two endpoints the Rust repair loop already speaks, so pointing
``NSYNTH_LOCAL_LLM_URL`` at this server needs ZERO Rust changes:

    GET  /v1/models            -> {"object":"list","data":[{"id":<model>,...}]}
    POST /v1/chat/completions  -> mlx_lm.generate() with the Mog logits_processor,
                                  returning the standard chat.completion shape
                                  (Rust reads choices[0].message.content).

The generation is constrained every decode step by
``mog_grammar.make_logits_processor`` (three layers: static drift-ban,
structural bracket-underflow mask, fence-keyed completion mask), which kills the
observed drift where the local model emits Rust ``let mut`` / Python instead of
Mog. Decoder-level guarantee, no training.

Design rules honoured here:
  * ``mlx_lm`` (and hence the model / GPU) is imported ONLY when we actually
    serve. ``--check`` proves the grammar<->server wiring with a FAKE vocab and
    numpy, WITHOUT loading the model or touching the GPU.
  * The model + tokenizer are loaded ONCE at startup; ``banned`` is computed
    ONCE over the real vocab.
  * A FRESH logits_processor is built per request (no prompt_len / structural
    state leak between requests).
  * The prompt is passed as an int-id LIST (avoids double-BOS: mlx_lm only
    BOS-encodes ``str`` prompts; a list is wrapped verbatim in ``mx.array``).
  * Single-threaded ``HTTPServer`` (the Rust repair loop is sequential); a global
    lock still guards generation so a future ThreadingHTTPServer stays safe
    (mlx generation is a single, non-re-entrant GPU stream).

mlx_lm contract (mlx_lm 0.31.2), verified against source:
  generate.py:307-322  generate_step(..., *, sampler, logits_processors=[(tokens,logits)->logits])
  generate.py:407      logits = logits[:, -1, :]                (shape (1, V))
  generate.py:408-416  processors run before the sampler; -inf reliably excludes
  generate.py:688-695  list/array prompt is NOT BOS-re-encoded (only str is)
  generate.py:756-762  generate(model, tokenizer, prompt, verbose=False, **kwargs) -> str
  sample_utils.py:10   make_sampler(temp=0.0, ...); temp==0 -> argmax

Usage:
    python3 scripts/mog_constrained_server.py --model <mlx-model-path> --port 8765
    python3 scripts/mog_constrained_server.py --check --model /dev/null   # no GPU
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

# mog_grammar lives next to this file; it imports ONLY stdlib at import time.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mog_grammar  # noqa: E402

# ---------------------------------------------------------------------------
# Serve-mode globals (populated once in serve(); untouched by --check).
# ---------------------------------------------------------------------------
_ARGS = None            # parsed CLI args
_MLX = None             # the mlx_lm module handle (imported in serve())
_MAKE_SAMPLER = None    # mlx_lm.sample_utils.make_sampler
_MODEL = None           # loaded model (once)
_TOKENIZER = None       # loaded TokenizerWrapper (once)
_BANNED = None          # frozenset of banned token ids (computed once)
_GEN_LOCK = threading.Lock()  # serialise generation (single GPU stream)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _generate_text(prompt_ids, temperature, max_tokens):
    """Run mlx_lm.generate with a FRESH Mog logits_processor. Returns str."""
    sampler = _MAKE_SAMPLER(temp=temperature)

    def _attempt(prompt_arg):
        # fresh processor per attempt -> no leaked prompt_len / structural state
        proc = mog_grammar.make_logits_processor(_TOKENIZER, _BANNED)
        return _MLX.generate(
            _MODEL,
            _TOKENIZER,
            prompt_arg,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=[proc],
            verbose=False,
        )

    with _GEN_LOCK:
        try:
            # preferred: pass the id LIST (no double-BOS)
            return _attempt(prompt_ids)
        except (TypeError, ValueError):
            # older/pickier mlx_lm may want an mx.array — wrap and retry.
            import mlx.core as mx  # local: only reached in serve mode
            return _attempt(mx.array(prompt_ids))


def _handle_chat(req):
    """Turn an OpenAI chat request dict into an OpenAI chat.completion dict."""
    messages = req.get("messages") or []
    model_name = req.get("model") or _ARGS.model
    temperature = float(req.get("temperature", 0.0) or 0.0)
    max_tokens = int(req.get("max_tokens", 512) or 512)

    # Apply the tokenizer chat template -> list[int] (return_dict=False forced by
    # the wrapper). add_generation_prompt=True so the assistant turn is primed.
    prompt_ids = _TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    prompt_ids = list(prompt_ids)

    text = _generate_text(prompt_ids, temperature, max_tokens)

    p = len(prompt_ids)
    try:
        c = len(_TOKENIZER.encode(text, add_special_tokens=False))
    except Exception:
        c = len(text.split())  # best-effort fallback; never fatal
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
        },
    }


def _models_payload():
    return {
        "object": "list",
        "data": [
            {
                "id": _ARGS.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mog",
            }
        ],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class MogHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(200, _models_payload())
        else:
            self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self):  # noqa: N802 (http.server API)
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw or b"{}")
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"bad request json: {exc}"}})
            return
        try:
            resp = _handle_chat(req)
        except Exception as exc:  # never crash the server on one bad request
            self._send_json(
                500,
                {"error": {"message": str(exc), "type": type(exc).__name__}},
            )
            return
        self._send_json(200, resp)

    def log_message(self, fmt, *args):  # keep stdout clean
        return


# ---------------------------------------------------------------------------
# Serve mode (loads the model ONCE)
# ---------------------------------------------------------------------------
def serve(args):
    global _ARGS, _MLX, _MAKE_SAMPLER, _MODEL, _TOKENIZER, _BANNED
    _ARGS = args

    import mlx_lm  # imported HERE so --check never needs mlx / the GPU
    from mlx_lm.sample_utils import make_sampler

    _MLX = mlx_lm
    _MAKE_SAMPLER = make_sampler

    print(f"[mog-server] loading model: {args.model}", flush=True)
    _MODEL, _TOKENIZER = mlx_lm.load(args.model)

    print("[mog-server] precomputing banned token ids over vocab ...", flush=True)
    vocab = _TOKENIZER.get_vocab()  # {token_str: id}
    _BANNED = mog_grammar.banned_token_ids(
        lambda i: _TOKENIZER.decode([i]), vocab.values()
    )
    print(
        f"[mog-server] {len(_BANNED)} banned ids over |V|={len(vocab)}; "
        f"listening on http://127.0.0.1:{args.port}",
        flush=True,
    )

    httpd = HTTPServer(("127.0.0.1", args.port), MogHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mog-server] shutting down", flush=True)
        httpd.server_close()


# ---------------------------------------------------------------------------
# --check mode (NO model, NO GPU): proves grammar<->server wiring with numpy.
# ---------------------------------------------------------------------------
class _FakeTok:
    """Minimal tokenizer stand-in: dict word<->id, decode, get_vocab."""

    def __init__(self, words):
        self._id2w = {i: w for i, w in enumerate(words)}
        self._w2id = {w: i for i, w in enumerate(words)}
        self.eos_token_id = None

    def get_vocab(self):
        return dict(self._w2id)

    def decode(self, ids):
        return "".join(self._id2w.get(int(i), "") for i in ids)


def check(args):
    """Verify grammar<->processor wiring with a fake vocab + numpy. No model."""
    import numpy as np

    words = [
        "let", "mut", "count", "fn", " ", "{", "}", "(", ")",
        "return", "1", ";", "->", "x",
    ]
    tok = _FakeTok(words)
    v = tok.get_vocab()

    banned = mog_grammar.banned_token_ids(lambda i: tok.decode([i]), v.values())
    assert v["let"] in banned, "'let' should be banned"
    assert v["mut"] in banned, "'mut' should be banned"
    assert v["count"] not in banned, "'count' must survive (word-boundary)"
    assert v["->"] not in banned, "'->' must never be banned"

    proc = mog_grammar.make_logits_processor(tok, banned, xp=np)
    V = len(words)
    neg = float("-inf")

    # First call: tokens == last prompt token only (prefill consumed the rest);
    # this captures prompt_len and applies the static ban + structural mask.
    t0 = np.array([v["fn"]])
    out0 = proc(t0, np.zeros((1, V), dtype=np.float32))
    assert float(out0.max()) != neg, "no-deadlock: a survivor must exist"
    assert float(out0[0, v["let"]]) == neg, "'let' logit must be -inf"
    assert float(out0[0, v["mut"]]) == neg, "'mut' logit must be -inf"

    # Second call: append a generated token to exercise the decode/feed path.
    t1 = np.array([v["fn"], v["count"]])
    out1 = proc(t1, np.zeros((1, V), dtype=np.float32))
    assert float(out1.max()) != neg, "no-deadlock after feed: survivor exists"

    print("OK: wiring valid (model NOT loaded)")
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Grammar-constrained Mog OpenAI-compatible server (mlx_lm)."
    )
    ap.add_argument("--model", required=True, help="path/name of the mlx model")
    ap.add_argument("--port", type=int, default=8765, help="listen port (default 8765)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify grammar<->server wiring with a fake vocab (no model/GPU), then exit",
    )
    args = ap.parse_args()

    if args.check:
        sys.exit(check(args))
    serve(args)


if __name__ == "__main__":
    main()
