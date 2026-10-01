"""Translate an existing .srt file, preserving timings (offline HF opus-mt)."""
from __future__ import annotations

import re
from pathlib import Path

import pysubs2

TAG_RE = re.compile(r"</?i>|</?b>|</?u>|\{[^}]*\}")


def clean(s: str) -> str:
    return TAG_RE.sub("", s).strip()


def split_dash(p: str) -> tuple[str, str]:
    m = re.match(r"^(\s*-\s*)(.*)$", p, flags=re.S)
    if m:
        return m.group(1), m.group(2)
    return "", p


def run(args) -> Path:
    src = Path(args.srt)
    if not src.exists():
        raise SystemExit(f"srt not found: {src}")
    model_id = args.model or f"Helsinki-NLP/opus-mt-{args.src}-{args.tgt}"
    print(f"[subsprint] loading translation model: {model_id}", flush=True)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()

    subs = pysubs2.load(str(src))
    seen: dict[str, None] = {}
    unique: list[str] = []
    for ev in subs:
        for p in re.split(r"\\N|\n", ev.text):
            _, core = split_dash(p)
            c = clean(core)
            if c and c not in seen:
                seen[c] = None
                unique.append(c)
    print(f"[subsprint] translating {len(unique)} unique lines ...", flush=True)

    out: dict[str, str] = {}
    B = args.batch_size or 32
    with torch.no_grad():
        for i in range(0, len(unique), B):
            batch = unique[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=128)
            gen = model.generate(**enc, max_length=128, num_beams=1)
            for s, t in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                out[s] = t
            if i % 320 == 0:
                print(f"[subsprint] {i}/{len(unique)}", flush=True)

    def tr_line(p: str) -> str:
        dash, core = split_dash(p)
        italic = "<i>" in core
        c = clean(core)
        if not c:
            return p
        t = out.get(c, c)
        return f"{dash}<i>{t}</i>" if italic else f"{dash}{t}"

    for ev in subs:
        parts = re.split(r"\\N|\n", ev.text)
        ev.text = "\\N".join(tr_line(p) for p in parts)

    if args.src == "it" and args.tgt == "en":
        for e in subs:
            t = e.text
            t = re.sub(r"\[uomo\]", "[man]", t, flags=re.I)
            t = re.sub(r"\[donna\]", "[woman]", t, flags=re.I)
            e.text = t

    dst = Path(args.output) if args.output else src.with_suffix(f".{args.tgt}.srt")
    subs.save(str(dst))
    print(f"[subsprint] saved: {dst} ({len(subs)} events)", flush=True)
    return dst
