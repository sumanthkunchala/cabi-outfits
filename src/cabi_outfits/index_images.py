# src/cabi_outfits/index_images.py
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

app = typer.Typer(help="FAISS index for image embeddings (cosine via inner product)")
CONSOLE = Console()


def _load_embeddings(emb_path: Path, ids_path: Path) -> Tuple[np.ndarray, List[str]]:
    embs = np.load(emb_path).astype("float32")
    with ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    if embs.shape[0] != len(ids):
        raise SystemExit(f"rows mismatch: {embs.shape[0]} embeddings vs {len(ids)} ids")
    # vectors are already L2-normalized in embed_images.py, but normalize again defensively
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms
    return embs, ids


@app.command("build")
def build(
    emb_path: str = typer.Argument(..., help="embeddings/image_clip_vitb32.npy"),
    ids_path: str = typer.Argument(..., help="embeddings/image_ids.json"),
    out_index: str = typer.Argument(..., help="embeddings/image_clip_vitb32.faiss"),
):
    emb_p = Path(emb_path)
    ids_p = Path(ids_path)
    out_p = Path(out_index)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    embs, ids = _load_embeddings(emb_p, ids_p)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine when vectors are unit length
    index.add(embs)

    faiss.write_index(index, str(out_p))

    t = Table(title="FAISS Build", show_lines=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Vectors", str(len(ids)))
    t.add_row("Dim", str(d))
    t.add_row("Index type", "IndexFlatIP (cosine)")
    t.add_row("Saved to", str(out_p))
    CONSOLE.print(t)


@app.command("selftest")
def selftest(
    items_parquet: str = typer.Argument(..., help="data/tmp/items_clean.parquet"),
    index_path: str = typer.Argument(..., help="embeddings/image_clip_vitb32.faiss"),
    ids_path: str = typer.Argument(..., help="embeddings/image_ids.json"),
    k: int = typer.Option(3, help="neighbors to show (excluding self)"),
    n: int = typer.Option(5, help="random probes"),
):
    items_df = pd.read_parquet(items_parquet)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    id_to_row = {pid: i for i, pid in enumerate(ids)}

    index = faiss.read_index(index_path)
    # load the vectors (for querying specific rows)
    embs, _ = _load_embeddings(Path(index_path).with_suffix(".npy").with_name(Path(index_path).stem + ".npy"), Path(ids_path))
    # Above assumes embeddings file alongside; be robust:
    try:
        embs = np.load("embeddings/image_clip_vitb32.npy").astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    except Exception:
        pass

    # sample n ids
    import random
    probes = random.sample(ids, min(n, len(ids)))

    for pid in probes:
        row_idx = id_to_row[pid]
        q = embs[row_idx : row_idx + 1]
        D, I = index.search(q, k + 1)  # includes self at rank 0
        neighbor_idxs = [i for i in I[0].tolist() if i != row_idx][:k]
        neighbor_scores = [float(s) for i, s in zip(I[0].tolist(), D[0].tolist()) if i != row_idx][:k]

        name = items_df.loc[items_df["product_id"].astype(str) == pid, "name"]
        subtype = items_df.loc[items_df["product_id"].astype(str) == pid, "subtype"]
        display_name = (name.iloc[0] if not name.empty else "") or ""
        display_subtype = (subtype.iloc[0] if not subtype.empty else "") or ""

        CONSOLE.rule(f"[bold]Probe {pid} — {display_name} ({display_subtype})")
        tbl = Table(show_lines=False)
        tbl.add_column("Rank")
        tbl.add_column("Product ID")
        tbl.add_column("Name")
        tbl.add_column("Subtype")
        tbl.add_column("Score", justify="right")

        for rank, (ni, sc) in enumerate(zip(neighbor_idxs, neighbor_scores), start=1):
            npid = ids[ni]
            nrow = items_df.loc[items_df["product_id"].astype(str) == npid]
            nname = nrow["name"].iloc[0] if not nrow.empty else ""
            ntype = nrow["subtype"].iloc[0] if not nrow.empty else ""
            tbl.add_row(str(rank), npid, nname, ntype, f"{sc:.3f}")
        CONSOLE.print(tbl)

    CONSOLE.rule("[bold green]Selftest complete")


if __name__ == "__main__":
    app()
