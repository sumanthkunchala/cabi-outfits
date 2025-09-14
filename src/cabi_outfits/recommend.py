# src/cabi_outfits/recommend.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import open_clip
import typer
from rich.console import Console
from rich.table import Table

try:
    from .taxonomy import Taxonomy, _clean_token
except ImportError:
    from taxonomy import Taxonomy, _clean_token

app = typer.Typer(help="Rule-based outfit recommendations over CLIP + MiniLM indices")
CONSOLE = Console()


# --------------------------
# Utility: prompt parsing
# --------------------------
def extract_prompt_signals(prompt: str, tx: Taxonomy) -> Tuple[List[str], List[str]]:
    """
    Very light NLP: extract target colors and occasions from the prompt.
    """
    tokens = [_clean_token(t) for t in prompt.split()]
    # Map synonyms -> canonical colors
    color_hits: List[str] = []
    for t in tokens:
        if t in tx.color_synonyms:
            color_hits.append(tx.color_synonyms[t])
        elif t in tx.palette:
            color_hits.append(t)
    # Dedup
    seen = set()
    color_hits = [c for c in color_hits if not (c in seen or seen.add(c))]

    # Occasions
    occ_hits: List[str] = []
    for t in tokens:
        t2 = tx.occasion_synonyms.get(t, t)
        if t2 in tx.allowed_occasions:
            occ_hits.append(t2)
    seen = set()
    occ_hits = [o for o in occ_hits if not (o in seen or seen.add(o))]

    # Special phrases
    lowered = prompt.lower()
    if "smart casual" in lowered and "smart casual" not in occ_hits:
        occ_hits.append("smart casual")
    if "work" in lowered and "work" not in occ_hits:
        occ_hits.append("work")
    if "dinner" in lowered and "dinner" not in occ_hits:
        occ_hits.append("dinner")
    if "weekend" in lowered and "weekend" not in occ_hits:
        occ_hits.append("weekend")
    if "travel" in lowered and "travel" not in occ_hits:
        occ_hits.append("travel")
    if "formal" in lowered and "formal" not in occ_hits:
        occ_hits.append("formal")
    if "party" in lowered and "party" not in occ_hits:
        occ_hits.append("party")

    return color_hits, occ_hits


# --------------------------
# Color harmony (coarse)
# --------------------------
_NEUTRALS = {"black", "white", "ivory", "cream", "gray", "silver", "beige", "tan", "camel", "brown", "gold"}
# rudimentary hue wheel (degrees)
_HUE: Dict[str, int] = {
    "navy": 230, "blue": 215, "teal": 190, "green": 140, "olive": 100, "yellow": 60,
    "orange": 30, "rust": 20, "red": 0, "burgundy": 350, "pink": 340, "blush": 345, "purple": 280,
}
def _hue(c: str) -> Optional[int]:
    return _HUE.get(c)

def _angle_diff(a: int, b: int) -> int:
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d

def color_harmony_score(colors: Sequence[str]) -> float:
    """
    Score 0..1: same/neutral/analogous/complementary get higher marks; clashes get lower.
    """
    cols = [c for c in colors if c]
    if not cols:
        return 0.5
    # If all neutral, fine
    if all(c in _NEUTRALS for c in cols):
        return 0.75
    # Remove neutrals for harmony calc
    chroma = [c for c in cols if c not in _NEUTRALS]
    if len(chroma) <= 1:
        return 0.8
    # pairwise hue diffs
    hs = [_hue(c) for c in chroma if _hue(c) is not None]
    if len(hs) < 2:
        return 0.6
    # average best-pair score
    scores = []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            diff = _angle_diff(hs[i], hs[j])
            if diff <= 30:      # analogous
                scores.append(0.9)
            elif 150 <= diff <= 210:  # complementary-ish
                scores.append(0.85)
            elif diff <= 60:    # near-analogous
                scores.append(0.75)
            else:
                scores.append(0.55)
    return float(sum(scores) / len(scores))


# --------------------------
# Similarity helpers
# --------------------------
@dataclass
class Catalog:
    items: pd.DataFrame
    img_ids: List[str]
    img_embs: np.ndarray
    id_to_row: Dict[str, int]

    @staticmethod
    def load(items_parquet: Path, img_ids_path: Path, img_emb_path: Path) -> "Catalog":
        items = pd.read_parquet(items_parquet)
        img_ids = json.loads(img_ids_path.read_text(encoding="utf-8"))
        img_embs = np.load(img_emb_path).astype("float32")
        norms = np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-12
        img_embs = img_embs / norms
        id_to_row = {pid: i for i, pid in enumerate(img_ids)}
        return Catalog(items=items, img_ids=img_ids, img_embs=img_embs, id_to_row=id_to_row)

    def sim(self, a: str, b: str) -> float:
        ia, ib = self.id_to_row.get(a), self.id_to_row.get(b)
        if ia is None or ib is None:
            return 0.0
        va, vb = self.img_embs[ia], self.img_embs[ib]
        return float(np.dot(va, vb))


def avg_pairwise_sim(cat: Catalog, pids: Sequence[str]) -> float:
    vecs = []
    for pid in pids:
        idx = cat.id_to_row.get(pid)
        if idx is not None:
            vecs.append(cat.img_embs[idx])
    if len(vecs) < 2:
        return 0.6
    V = np.stack(vecs, axis=0)
    S = V @ V.T
    # exclude diagonal
    m = (np.sum(S) - np.trace(S)) / (S.shape[0] * (S.shape[0] - 1))
    return float(m)


def occasions_score(cat: Catalog, pids: Sequence[str], desired: Sequence[str]) -> float:
    if not desired:
        return 0.6
    desired = set(desired)
    hits = 0; total = 0
    for pid in pids:
        row = cat.items.loc[cat.items["product_id"].astype(str) == pid]
        if row.empty:
            continue
        occasions_data = row.iloc[0].get("occasions")
        if occasions_data is not None and hasattr(occasions_data, '__len__') and len(occasions_data) > 0:
            occasions_list = [str(o) for o in occasions_data]
        else:
            occasions_list = []
        occ = set(occasions_list).intersection(desired)
        total += 1
        if occ:
            hits += 1
    if total == 0:
        return 0.6
    return 0.5 + 0.5 * (hits / total)  # 0.5..1.0


def palette_score(cat: Catalog, pids: Sequence[str], desired_colors: Sequence[str]) -> float:
    cols: List[str] = []
    for pid in pids:
        row = cat.items.loc[cat.items["product_id"].astype(str) == pid]
        if row.empty: 
            continue
        colors_data = row.iloc[0].get("colors")
        if colors_data is not None and hasattr(colors_data, '__len__') and len(colors_data) > 0:
            cols.extend([str(c) for c in colors_data])
        else:
            cols.extend([])
    base = color_harmony_score(cols)
    if desired_colors:
        got = any(c in cols for c in desired_colors)
        base += 0.1 if got else 0.0
    return min(1.0, base)


def diversity_penalty(cat: Catalog, pids: Sequence[str]) -> float:
    """
    Lower if we have multiple loud patterns.
    """
    loud = {"stripe", "plaid", "check", "houndstooth", "floral", "animal", "polka dot", "geometric", "abstract", "print"}
    count = 0
    for pid in pids:
        row = cat.items.loc[cat.items["product_id"].astype(str) == pid]
        if row.empty:
            continue
        patterns_data = row.iloc[0].get("patterns")
        if patterns_data is not None and hasattr(patterns_data, '__len__') and len(patterns_data) > 0:
            patterns_list = [str(p) for p in patterns_data]
        else:
            patterns_list = []
        pats = set(patterns_list)
        if pats.intersection(loud):
            count += 1
    if count <= 1:
        return 1.0
    if count == 2:
        return 0.9
    return 0.8


def outfit_score(cat: Catalog, pids: Sequence[str], desired_colors: Sequence[str], desired_occasions: Sequence[str]) -> float:
    s_sim = avg_pairwise_sim(cat, pids)              # 0..1-ish (cosine avg)
    s_occ = occasions_score(cat, pids, desired_occasions)   # 0.5..1
    s_pal = palette_score(cat, pids, desired_colors)        # 0.5..1
    s_div = diversity_penalty(cat, pids)                    # 0.8..1
    return 0.35*s_sim + 0.30*s_occ + 0.20*s_pal + 0.15*s_div


# --------------------------
# Retrieval & assembly
# --------------------------
def clip_text_to_index(prompt: str, index: faiss.Index, model, device) -> Tuple[List[int], List[float]]:
    with torch.no_grad():
        tok = open_clip.tokenize([prompt]).to(device)
        tfeat = model.encode_text(tok)
        tfeat = torch.nn.functional.normalize(tfeat, p=2, dim=-1).cpu().numpy().astype("float32")
    D, I = index.search(tfeat, 200)  # top-200 candidates
    return I[0].tolist(), D[0].tolist()


def assemble_outfits(
    prompt: str,
    cat: Catalog,
    faiss_index: faiss.Index,
    img_ids: List[str],
    tx: Taxonomy,
    must_include: Optional[List[str]] = None,
    k: int = 8,
) -> List[Dict]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    clip_model.eval()

    cand_idxs, cand_scores = clip_text_to_index(prompt, faiss_index, clip_model, device)
    # Map to product_ids and filter to things we actually have in the catalog with categories
    cands: List[Tuple[str, float, str]] = []
    for idx, sc in zip(cand_idxs, cand_scores):
        if idx < 0 or idx >= len(img_ids):
            continue
        pid = img_ids[idx]
        row = cat.items.loc[cat.items["product_id"].astype(str) == pid]
        if row.empty:
            continue
        category = row.iloc[0].get("category")
        if category not in {"top", "bottom", "dress", "outerwear", "accessory"}:
            continue
        cands.append((pid, float(sc), category))

    # Partition
    part: Dict[str, List[Tuple[str, float]]] = {c: [] for c in ["top", "bottom", "dress", "outerwear", "accessory"]}
    for pid, sc, catg in cands:
        part[catg].append((pid, sc))

    # Keep top-N per bucket to control combinatorics
    topN = {"top": 20, "bottom": 20, "dress": 20, "outerwear": 10, "accessory": 20}
    for kcat in part:
        part[kcat] = part[kcat][: topN.get(kcat, 20)]

    # Prompt signals
    desired_colors, desired_occasions = extract_prompt_signals(prompt, tx)

    outfits: List[Dict] = []

    # Bases: (dress) or (top+bottom)
    # --- Dress base ---
    for pid, _ in part["dress"]:
        base = [pid]
        # Optional add outerwear/accessory
        ow = part["outerwear"][:5]
        acc = part["accessory"][:8]
        candidates_sets = []

        # maybe with outerwear
        for opid, _ in ow[:3]:
            candidates_sets.append([pid, opid])
        # maybe with accessories (0,1,2)
        for a1 in acc[:3]:
            candidates_sets.append([pid, a1[0]])
        for i in range(min(2, len(acc))):
            for j in range(i+1, min(5, len(acc))):
                candidates_sets.append([pid, acc[i][0], acc[j][0]])

        for combo in ([[pid]] + candidates_sets):
            if must_include and not set(must_include).issubset(set(combo)):
                continue
            score = outfit_score(cat, combo, desired_colors, desired_occasions)
            outfits.append({"items": combo, "score": score})

    # --- Top+Bottom base ---
    for tpid, _ in part["top"][:20]:
        for bpid, _ in part["bottom"][:20]:
            base = [tpid, bpid]
            # Optional layers
            ow = part["outerwear"][:5]
            acc = part["accessory"][:8]

            combos = [base]
            for opid, _ in ow[:3]:
                combos.append([tpid, bpid, opid])
            # accessories
            for i in range(min(2, len(acc))):
                combos.append([tpid, bpid, acc[i][0]])
            if len(acc) >= 2:
                combos.append([tpid, bpid, acc[0][0], acc[1][0]])

            for combo in combos:
                if must_include and not set(must_include).issubset(set(combo)):
                    continue
                score = outfit_score(cat, combo, desired_colors, desired_occasions)
                outfits.append({"items": combo, "score": score})

    # Sort & de-duplicate by item-set
    def sig(o: Dict) -> Tuple[str, ...]:
        return tuple(sorted(o["items"]))
    seen = set()
    uniq: List[Dict] = []
    for o in sorted(outfits, key=lambda x: x["score"], reverse=True):
        s = sig(o)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(o)
        if len(uniq) >= k:
            break

    # Attach metadata & reasons
    results: List[Dict] = []
    for o in uniq:
        items_meta = []
        colors: List[str] = []
        cats = []
        for pid in o["items"]:
            row = cat.items.loc[cat.items["product_id"].astype(str) == pid]
            if row.empty: 
                continue
            r = row.iloc[0]
            # Safely handle array-like fields
            colors_data = r.get("colors")
            if colors_data is not None and hasattr(colors_data, '__len__') and len(colors_data) > 0:
                colors_list = [str(c) for c in colors_data]
            else:
                colors_list = []
            
            patterns_data = r.get("patterns")
            if patterns_data is not None and hasattr(patterns_data, '__len__') and len(patterns_data) > 0:
                patterns_list = [str(p) for p in patterns_data]
            else:
                patterns_list = []
            
            occasions_data = r.get("occasions")
            if occasions_data is not None and hasattr(occasions_data, '__len__') and len(occasions_data) > 0:
                occasions_list = [str(o) for o in occasions_data]
            else:
                occasions_list = []
            
            items_meta.append({
                "product_id": pid,
                "name": r.get("name"),
                "subtype": r.get("subtype"),
                "category": r.get("category"),
                "colors": colors_list,
                "patterns": patterns_list,
                "occasions": occasions_list,
            })
            # Safely handle colors for palette building
            colors_data_pal = r.get("colors")
            if colors_data_pal is not None and hasattr(colors_data_pal, '__len__') and len(colors_data_pal) > 0:
                colors.extend([str(c) for c in colors_data_pal])
            else:
                colors.extend([])
            cats.append(r.get("category"))
        pal = list(dict.fromkeys(colors))[:3]
        base = "dress" if ("dress" in cats and "top" not in cats and "bottom" not in cats) else "top+bottom"
        results.append({
            "score": round(float(o["score"]), 3),
            "palette": pal,
            "base": base,
            "items": items_meta
        })
    return results


@app.command()
def recommend(
    prompt: str = typer.Argument(..., help="e.g., 'smart casual dinner outfit in fall tones'"),
    items_parquet: str = typer.Option("data/tmp/items_clean.parquet"),
    img_ids: str = typer.Option("embeddings/image_ids.json"),
    img_embs: str = typer.Option("embeddings/image_clip_vitb32.npy"),
    img_index: str = typer.Option("embeddings/image_clip_vitb32.faiss"),
    must_include: Optional[str] = typer.Option(None, help="comma-separated product_ids that must appear"),
    k: int = typer.Option(8, help="number of outfits to return"),
):
    tx = Taxonomy.load()
    cat = Catalog.load(Path(items_parquet), Path(img_ids), Path(img_embs))
    index = faiss.read_index(img_index)
    must = [m.strip() for m in (must_include.split(",") if must_include else []) if m.strip()]
    outfits = assemble_outfits(prompt, cat, index, cat.img_ids, tx, must_include=must or None, k=k)

    # Pretty print
    CONSOLE.rule(f'[bold]Outfits for: "{prompt}"')
    for i, o in enumerate(outfits, 1):
        t = Table(title=f"Outfit {i} — score {o['score']}", show_lines=False)
        t.add_column("Product ID"); t.add_column("Name"); t.add_column("Subtype"); t.add_column("Category"); t.add_column("Colors"); t.add_column("Occasions")
        for it in o["items"]:
            t.add_row(str(it["product_id"]), it["name"] or "", it["subtype"] or "", it["category"] or "", ", ".join(it["colors"]), ", ".join(it["occasions"]))
        CONSOLE.print(t)
        CONSOLE.print(f"Palette guess: {', '.join(o['palette'])} | Base: {o['base']}")
        CONSOLE.rule()
    CONSOLE.print(f"Returned {len(outfits)} outfits.")
    CONSOLE.rule("[bold green]Done")


if __name__ == "__main__":
    app()
