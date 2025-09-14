# src/cabi_outfits/index_text.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

app = typer.Typer(help="FAISS index + prompt search for MiniLM text embeddings")
CONSOLE = Console()


def _load(emb_path: Path, ids_path: Path) -> Tuple[np.ndarray, List[str]]:
    embs = np.load(emb_path).astype("float32")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if embs.shape[0] != len(ids):
        raise SystemExit(f"rows mismatch: {embs.shape[0]} embeddings vs {len(ids)} ids")
    # re-normalize defensively
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms
    return embs, ids


@app.command("build")
def build(
    emb_path: str = typer.Argument(..., help="embeddings/text_minilm.npy"),
    ids_path: str = typer.Argument(..., help="embeddings/text_ids.json"),
    out_index: str = typer.Argument(..., help="embeddings/text_minilm.faiss"),
):
    emb_p, ids_p, out_p = Path(emb_path), Path(ids_path), Path(out_index)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    embs, ids = _load(emb_p, ids_p)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine on unit vectors
    index.add(embs)
    faiss.write_index(index, str(out_p))

    t = Table(title="Text FAISS Build", show_lines=True)
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Vectors", str(len(ids)))
    t.add_row("Dim", str(d))
    t.add_row("Index type", "IndexFlatIP (cosine)")
    t.add_row("Saved to", str(out_p))
    CONSOLE.print(t)


@app.command("query")
def query(
    q: str = typer.Argument(..., help="free-text prompt"),
    index_path: str = typer.Argument(..., help="embeddings/text_minilm.faiss"),
    ids_path: str = typer.Argument(..., help="embeddings/text_ids.json"),
    items_parquet: str = typer.Argument(..., help="data/tmp/items_clean.parquet"),
    k: int = typer.Option(8, help="top-K results"),
):
    # embed prompt
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # load index + ids
    index = faiss.read_index(index_path)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    items = pd.read_parquet(items_parquet)

    D, I = index.search(q_emb, k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    tbl = Table(title=f'Query: "{q}"', show_lines=False)
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
