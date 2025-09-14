# src/cabi_outfits/ui_gradio.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import faiss

from .recommend import assemble_outfits, Catalog
from .taxonomy import Taxonomy

APP_ROOT = Path(__file__).resolve().parents[2]  # project root
ITEMS_PARQUET = APP_ROOT / "data" / "tmp" / "items_clean.parquet"
IMG_IDS = APP_ROOT / "embeddings" / "image_ids.json"
IMG_EMBS = APP_ROOT / "embeddings" / "image_clip_vitb32.npy"
IMG_INDEX = APP_ROOT / "embeddings" / "image_clip_vitb32.faiss"
IMG_DIR = APP_ROOT / "data" / "images"

# ---------- helpers ----------
def _load_globals():
    tx = Taxonomy.load()
    cat = Catalog.load(ITEMS_PARQUET, IMG_IDS, IMG_EMBS)
    index = faiss.read_index(str(IMG_INDEX))
    return tx, cat, index

def _safe_open(pid: str, height: int = 420) -> Image.Image:
    p = IMG_DIR / f"{pid}.jpg"
    if not p.exists():
        # placeholder
        img = Image.new("RGB", (height, height), (240, 240, 240))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"{pid}\n(no image)", fill=(0, 0, 0))
        return img
    img = Image.open(p).convert("RGB")
    # resize by height, keep aspect
    w, h = img.size
    new_w = int(w * (height / float(h)))
    return img.resize((new_w, height))

def _render_strip(pids: List[str], captions: List[str]) -> Image.Image:
    # Load images, concat horizontally, draw captions below each
    imgs = [_safe_open(pid, height=420) for pid in pids]
    gaps = 20
    total_w = sum(im.size[0] for im in imgs) + gaps * (len(imgs) - 1)
    bar_h = 48
    H = 420 + bar_h
    canvas = Image.new("RGB", (total_w, H), (255, 255, 255))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for im, cap in zip(imgs, captions):
        canvas.paste(im, (x, 0))
        # caption strip
        draw.rectangle([x, 420, x + im.size[0], 420 + bar_h], fill=(250, 250, 250))
        draw.text((x + 8, 420 + 12), cap[:60], fill=(0, 0, 0))
        x += im.size[0] + gaps
    return canvas

# ---------- UI action ----------
def recommend_ui(prompt: str, k: int, must_include_csv: str) -> Tuple[List[Tuple[Image.Image, str]], str]:
    tx, cat, index = _load_globals()
    must = [s.strip() for s in (must_include_csv or "").split(",") if s.strip()]
    outfits = assemble_outfits(
        prompt=prompt,
        cat=cat,
        faiss_index=index,
        img_ids=cat.img_ids,
        tx=tx,
        must_include=must or None,
        k=int(k),
    )
    gallery: List[Tuple[Image.Image, str]] = []
    md_lines: List[str] = []
    for i, o in enumerate(outfits, 1):
        pids = [it["product_id"] for it in o["items"]]
        caps = [f'{it["category"]}: {it["name"] or ""}' for it in o["items"]]
        img = _render_strip(pids, caps)
        caption = f'#{i}  score={o["score"]:.3f}  base={o["base"]}  palette={", ".join(o["palette"])}'
        gallery.append((img, caption))

        # markdown details
        md_lines.append(f"### Outfit {i} — score {o['score']:.3f}, base: {o['base']}, palette: {', '.join(o['palette'])}")
        for it in o["items"]:
            md_lines.append(f"- **{it['product_id']}** — {it['name']} *( {it['category']}/{it['subtype']} )*  • colors: {', '.join(it['colors'])}  • occasions: {', '.join(it['occasions'])}")
        md_lines.append("")

    if not gallery:
        return [], "_No outfits found. Try a broader prompt or remove must_include._"
    return gallery, "\n".join(md_lines)

def main():
    with gr.Blocks(title="Cabi Outfit Recommender") as demo:
        gr.Markdown("# Cabi Outfit Recommender (MVP)\nType a style prompt; optionally pin items by product_id.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="smart casual dinner outfit in fall tones")
        with gr.Row():
            k = gr.Slider(1, 8, value=4, step=1, label="How many outfits?")
            must = gr.Textbox(label="Must include (comma-separated product_id)", placeholder="e.g., 461203,462286")
        submit = gr.Button("Recommend", variant="primary")
        gallery = gr.Gallery(label="Outfits", show_label=True, columns=1)
        details = gr.Markdown()

        submit.click(fn=recommend_ui, inputs=[prompt, k, must], outputs=[gallery, details])

    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
