# src/cabi_outfits/embed_images.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import typer
from tqdm import tqdm
import open_clip

app = typer.Typer(help="Encode product images with OpenCLIP ViT-B/32")


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_model() -> Tuple[torch.nn.Module, callable]:
    # ViT-B/32 pretrained on LAION-2B (good general style semantics)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=_device(),
    )
    model.eval()
    return model, preprocess


def _iter_batches(paths: List[Path], batch_size: int):
    for i in range(0, len(paths), batch_size):
        yield paths[i : i + batch_size]


@app.command()
def run(
    items_parquet: str = typer.Argument(..., help="data/tmp/items_clean.parquet"),
    images_dir: str = typer.Option("data/images", help="Directory with {product_id}.jpg"),
    out_npy: str = typer.Option("embeddings/image_clip_vitb32.npy", help="Output .npy for embeddings"),
    out_ids: str = typer.Option("embeddings/image_ids.json", help="Output JSON array of product_ids aligned with rows"),
    batch_size: int = typer.Option(32, help="Batch size (reduce if low RAM)"),
):
    items_parquet = Path(items_parquet)
    images_dir = Path(images_dir)
    out_npy = Path(out_npy)
    out_ids = Path(out_ids)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_ids.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(items_parquet)
    if "product_id" not in df.columns:
        raise SystemExit("items_clean.parquet missing 'product_id' column")

    # Keep only rows with an existing image file
    ids: List[str] = []
    img_paths: List[Path] = []
    for pid in df["product_id"].astype(str).tolist():
        p = images_dir / f"{pid}.jpg"
        if p.exists():
            ids.append(pid)
            img_paths.append(p)

    if not img_paths:
        raise SystemExit("No images found to embed. Did you run the downloader?")

    model, preprocess = _load_model()
    dev = _device()

    # Probe one image to get embedding dim
    with torch.no_grad():
        probe_img = preprocess(Image.open(img_paths[0]).convert("RGB")).unsqueeze(0).to(dev)
        d = model.encode_image(probe_img).shape[-1]

    embs = np.zeros((len(img_paths), d), dtype="float32")

    row = 0
    with torch.no_grad():
        for batch_paths in tqdm(_iter_batches(img_paths, batch_size), total=(len(img_paths) + batch_size - 1)//batch_size, desc="Embedding images"):
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    # fallback: zero vector (we'll filter later if needed)
                    imgs.append(torch.zeros_like(preprocess(Image.new("RGB", (224, 224)))))
            x = torch.stack(imgs, dim=0).to(dev)

            feats = model.encode_image(x)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)  # unit vectors
            n = feats.shape[0]
            embs[row : row + n] = feats.cpu().float().numpy()
            row += n

    # Save outputs
    np.save(out_npy, embs)
    with out_ids.open("w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"Saved embeddings: {out_npy} shape={embs.shape}")
    print(f"Saved ids map:    {out_ids} count={len(ids)}")
    print(f"Device used:      {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("Done.")


if __name__ == "__main__":
    app()
