# src/cabi_outfits/query_clip.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
import typer
import torch
import open_clip
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="CLIP text → image retrieval over the image FAISS index")
CONSOLE = Console()


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_clip():
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=_device(),
    )
    model.eval()
    return model


@app.command()
def query(
    prompt: str = typer.Argument(..., help="Free-text user prompt"),
    index_path: str = typer.Option("embeddings/image_clip_vitb32.faiss", help="FAISS index path"),
    ids_path: str = typer.Option("embeddings/image_ids.json", help="JSON list of product_ids aligned to index rows"),
    items_parquet: str = typer.Option("data/tmp/items_clean.parquet", help="Catalog parquet for metadata"),
    k: int = typer.Option(8, help="Top-K results"),
):
    # Load index + ids
    index = faiss.read_index(index_path)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    items = pd.read_parquet(items_parquet)

    # Encode prompt with CLIP text tower (same checkpoint as image embeddings)
    model = _load_clip()
    with torch.no_grad():
        tok = open_clip.tokenize([prompt]).to(_device())
        tfeat = model.encode_text(tok)
        tfeat = torch.nn.functional.normalize(tfeat, p=2, dim=-1).cpu().numpy().astype("float32")

    # Search cosine (inner product on normalized vectors)
    D, I = index.search(tfeat, k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    # Print results
    tbl = Table(title=f'CLIP Query: "{prompt}"', show_lines=False)
    tbl.add_column("Rank"); tbl.add_column("Product ID"); tbl.add_column("Name")
    tbl.add_column("Subtype"); tbl.add_column("Category"); tbl.add_column("Score", justify="right")

    for rank, (ii, sc) in enumerate(zip(idxs, scores), start=1):
        pid = ids[ii]
        row = items.loc[items["product_id"].astype(str) == pid]
        name = row["name"].iloc[0] if not row.empty else ""
        subtype = row["subtype"].iloc[0] if not row.empty else ""
        category = row["category"].iloc[0] if not row.empty else ""
        tbl.add_row(str(rank), pid, name, subtype, category, f"{float(sc):.3f}")

    CONSOLE.print(tbl)
    CONSOLE.rule("[bold green]Done")


if __name__ == "__main__":
    app()
