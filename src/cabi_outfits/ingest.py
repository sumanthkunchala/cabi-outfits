# src/cabi_outfits/ingest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table
import typer

from .taxonomy import Taxonomy, _split_tokens, _clean_token
from .schema import Item

app = typer.Typer(help="Ingest & normalize the product catalog")

CONSOLE = Console()

# Column name candidates (lowercased)
CANDIDATES = {
    "product_id": ["style product id", "product_id", "product id", "sku", "id"],
    "name": ["name", "product name", "title"],
    "season": ["season"],
    "subtype": ["type", "subcategory", "subtype", "category"],  # we remap to canonical
    "price": ["price", "retail price", "list price"],
    "image_url": ["image", "image url", "image_url", "image link", "imageurl"],
    "color": ["color", "colors", "colour"],
    "occasion": ["occasion", "occasions"],
    "tags": ["tags", "keywords", "attributes"],
}

PATTERN_HINTS = ["stripe", "striped", "plaid", "check", "houndstooth", "floral",
                 "animal", "leopard", "zebra", "snakeskin", "polka dot", "dots",
                 "geometric", "abstract", "print", "textured", "rib", "ribbed", "jacquard", "cable"]


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.strip().lower(): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for key, options in CANDIDATES.items():
        for opt in options:
            if opt in cols:
                mapping[key] = cols[opt]
                break
    missing = [k for k in ["product_id", "name"] if k not in mapping]
    if missing:
        raise RuntimeError(f"Missing required columns in Excel: {missing}. "
                           f"Found columns: {list(df.columns)}")
    return mapping


def _extract_patterns_from_text(texts: List[str], taxonomy: Taxonomy) -> List[str]:
    found = set()
    for t in texts:
        for tok in _split_tokens(t or ""):
            tok = _clean_token(tok)
            if not tok:
                continue
            # map normalized hints
            if tok in taxonomy.pattern_norm:
                tok = taxonomy.pattern_norm[tok]
            # direct allow-list
            if tok in taxonomy.allowed_patterns:
                found.add(tok)
            # heuristic
            for hint in PATTERN_HINTS:
                if hint in tok:
                    norm = taxonomy.pattern_norm.get(hint, hint)
                    if norm in taxonomy.allowed_patterns:
                        found.add(norm)
    return list(found)


def normalize_row(row: pd.Series, cols: Dict[str, str], tx: Taxonomy) -> Tuple[Item | None, str | None]:
    # Basic fields
    product_id = str(row[cols["product_id"]]).strip()
    name = str(row[cols["name"]]).strip()

    if not product_id or not name:
        return None, "missing product_id or name"

    season = str(row[cols["season"]]).strip() if "season" in cols and pd.notna(row[cols["season"]]) else None
    subtype_raw = str(row[cols["subtype"]]).strip() if "subtype" in cols and pd.notna(row[cols["subtype"]]) else None
    category = tx.canonicalize_category(subtype_raw) if subtype_raw else None

    price = None
    if "price" in cols and pd.notna(row[cols["price"]]):
        try:
            price = float(str(row[cols["price"]]).replace("$", "").replace(",", ""))
        except Exception:
            price = None

    image_url = None
    if "image_url" in cols and pd.notna(row[cols["image_url"]]):
        image_url = str(row[cols["image_url"]]).strip()

    # Colors and patterns
    raw_color = str(row[cols["color"]]) if "color" in cols and pd.notna(row[cols["color"]]) else ""
    color_tokens = _split_tokens(raw_color)
    colors = tx.normalize_colors(color_tokens)

    # Patterns from color/tags/occasion text blobs
    raw_tags = str(row[cols["tags"]]) if "tags" in cols and pd.notna(row[cols["tags"]]) else ""
    raw_occ = str(row[cols["occasion"]]) if "occasion" in cols and pd.notna(row[cols["occasion"]]) else ""
    patterns = list(set(_extract_patterns_from_text([raw_color, raw_tags, raw_occ], tx)))

    occasions = tx.normalize_occasions([raw_occ])

    # Tags: union of tokens from tags + name keywords
    tags_tokens = list({_clean_token(t) for t in (_split_tokens(raw_tags) + name.lower().split()) if _clean_token(t)})

    item = Item(
        product_id=product_id,
        name=name,
        season=season,
        category=category,
        subtype=subtype_raw,
        price=price,
        image_url=image_url,
        colors=colors,
        patterns=patterns,
        occasions=occasions,
        tags=tags_tokens,
    )
    # Soft warning if no image_url — allowed for now; we'll try to download later and surface failures
    if not image_url:
        return item, "missing image_url"
    return item, None


@app.command()
def ingest(excel_path: str = typer.Argument(..., help="Path to products.xlsx"),
           out_dir: str = typer.Option("data/tmp", help="Output directory for normalized data")):
    """
    Read the Excel catalog, normalize fields using taxonomy.yaml, and emit:
      - data/tmp/items_clean.parquet
      - data/tmp/rejects.csv
    """
    excel_path = Path(excel_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tx = Taxonomy.load()

    CONSOLE.rule("[bold]Ingest start")
    CONSOLE.print(f"Reading: {excel_path}")
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]

    cols = _resolve_columns(df)

    goods: List[dict] = []
    rejects: List[dict] = []

    for _, row in df.iterrows():
        try:
            item, warn = normalize_row(row, cols, tx)
            if item is None:
                rejects.append({"reason": "invalid row", **{k: row.get(v, None) for k, v in cols.items()}})
                continue
            goods.append(item.to_dict())
            if warn:
                rejects.append({"reason": warn, **item.to_dict()})
        except Exception as e:
            rejects.append({"reason": f"exception: {e}", **{k: row.get(v, None) for k, v in cols.items()}})

    good_df = pd.DataFrame(goods)
    rej_df = pd.DataFrame(rejects)

    good_path = out / "items_clean.parquet"
    rej_path = out / "rejects.csv"

    good_df.to_parquet(good_path, index=False)
    rej_df.to_csv(rej_path, index=False)

    # Pretty summary
    t = Table(title="Ingest Summary", show_lines=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Input rows", str(len(df)))
    t.add_row("Valid items", str(len(good_df)))
    t.add_row("Rejects / Warnings", str(len(rej_df)))
    t.add_row("Output (clean)", str(good_path))
    t.add_row("Output (rejects)", str(rej_path))
    CONSOLE.print(t)
    CONSOLE.rule("[bold green]Done")


if __name__ == "__main__":
    app()
