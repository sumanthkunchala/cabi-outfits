# src/cabi_outfits/download_images.py
from __future__ import annotations

import io
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.console import Console
from rich.table import Table
import typer
from tqdm import tqdm


app = typer.Typer(help="Download product images and create a manifest")
CONSOLE = Console()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CabiOutfitsBot/1.0; +https://example.com)"
}


def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _to_rgb(img: Image.Image) -> Image.Image:
    # Convert to RGB (flatten alpha if present)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


@dataclass
class DownloadResult:
    product_id: str
    url: Optional[str]
    status: str  # "ok" | "skip" | "error"
    reason: Optional[str]
    width: Optional[int]
    height: Optional[int]
    sha256: Optional[str]
    saved_path: Optional[str]
    content_type: Optional[str]
    bytes: Optional[int]


def _download_one(session: requests.Session, row: Dict, out_dir: Path, timeout: float = 15.0) -> DownloadResult:
    pid = str(row.get("product_id", "")).strip()
    url = str(row.get("image_url") or "").strip()
    if not pid:
        return DownloadResult(pid, url, "error", "missing product_id", None, None, None, None, None, None)
    if not url:
        return DownloadResult(pid, url, "skip", "missing image_url", None, None, None, None, None, None)

    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout, stream=True)
    except Exception as e:
        return DownloadResult(pid, url, "error", f"request failed: {e}", None, None, None, None, None, None)

    if resp.status_code >= 400:
        return DownloadResult(pid, url, "error", f"http {resp.status_code}", None, None, None, None, None, None)

    # read up to ~20MB
    content = resp.content
    if not content or len(content) < 1024:
        return DownloadResult(pid, url, "error", "too small / empty", None, None, None, None, resp.headers.get("Content-Type"), len(content) if content else 0)

    # verify it's an image
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception as e:
        return DownloadResult(pid, url, "error", f"not an image: {e}", None, None, None, None, resp.headers.get("Content-Type"), len(content))

    img = _to_rgb(img)
    w, h = img.size

    # hash original bytes
    sha = hashlib.sha256(content).hexdigest()

    out_path = out_dir / f"{pid}.jpg"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # save as JPEG
        img.save(out_path, format="JPEG", quality=90, optimize=True)
    except Exception as e:
        return DownloadResult(pid, url, "error", f"save failed: {e}", w, h, sha, None, resp.headers.get("Content-Type"), len(content))

    return DownloadResult(pid, url, "ok", None, w, h, sha, str(out_path), resp.headers.get("Content-Type"), len(content))


@app.command()
def run(clean_parquet: str = typer.Argument(..., help="Path to data/tmp/items_clean.parquet"),
        out_dir: str = typer.Option("data/images", help="Directory to save images"),
        manifest_out: str = typer.Option("data/tmp/images_manifest.csv", help="CSV manifest path"),
        max_workers: int = typer.Option(16, help="Parallel download workers")):
    """
    Download images listed in items_clean.parquet.
    Emits:
      - data/images/{product_id}.jpg
      - data/tmp/images_manifest.csv
    """
    clean_parquet = Path(clean_parquet)
    out_dir = Path(out_dir)
    manifest_out = Path(manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(clean_parquet)
    rows = df.to_dict(orient="records")

    session = _requests_session()

    results: List[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_download_one, session, r, out_dir) for r in rows]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            results.append(f.result())

    # Write manifest CSV
    recs = [r.__dict__ for r in results]
    mdf = pd.DataFrame(recs)
    mdf.to_csv(manifest_out, index=False)

    # Summary
    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skip")
    errors = sum(1 for r in results if r.status == "error")

    t = Table(title="Image Download Summary", show_lines=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Input items", str(len(rows)))
    t.add_row("Saved images", str(ok))
    t.add_row("Skipped (no URL)", str(skipped))
    t.add_row("Errors", str(errors))
    t.add_row("Manifest", str(manifest_out))
    CONSOLE.print(t)

    # Also dump a small JSONL for programmatic use
    jsonl_path = manifest_out.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for r in results:
            jf.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    CONSOLE.print(f"Wrote JSONL: {jsonl_path}")

    CONSOLE.rule("[bold green]Done")


if __name__ == "__main__":
    app()
