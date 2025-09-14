# src/cabi_outfits/taxonomy.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Set
import re
import yaml


_CLEANER = re.compile(r"[^a-z0-9 +\-]")


def _clean_token(s: str) -> str:
    """
    Lowercase, trim, and remove weird punctuation. Keep spaces and dashes.
    """
    s = (s or "").strip().lower()
    s = _CLEANER.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _split_tokens(s: str) -> List[str]:
    """
    Split on commas or slashes; fall back to whitespace. Returns cleaned tokens.
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
        return [_clean_token(t) for t in s if str(t).strip()]
    # commas / slashes are common delimiters in catalogs
    parts = re.split(r"[,/]", str(s))
    if len(parts) == 1:
        parts = re.split(r"\s*[|]\s*", str(s))  # support "a | b"
    return [_clean_token(p) for p in parts if _clean_token(p)]


class Taxonomy:
    def __init__(self, raw: Dict):
        self.raw = raw

        # Categories
        self.canonical_categories: Set[str] = set(
            (raw.get("categories", {}) or {}).get("canonical", [])
        )
        self.subtype_to_category: Dict[str, str] = {
            _clean_token(k): v for k, v in ((raw.get("categories", {}) or {}).get("subtype_to_category", {}) .items())
        }

        # Colors
        colors = raw.get("colors", {}) or {}
        self.palette: Set[str] = set(colors.get("palette", []))
        self.color_synonyms: Dict[str, str] = { _clean_token(k): _clean_token(v) for k, v in (colors.get("synonyms", {}) or {}).items() }
        self.max_dominant: int = int(colors.get("max_dominant", 3))

        # Patterns
        pats = raw.get("patterns", {}) or {}
        self.allowed_patterns: Set[str] = set(pats.get("allowed", []))
        self.pattern_norm: Dict[str, str] = { _clean_token(k): _clean_token(v) for k, v in (pats.get("normalize", {}) or {}).items() }

        # Occasions
        occ = raw.get("occasions", {}) or {}
        self.allowed_occasions: Set[str] = set(occ.get("allowed", []))
        self.occasion_synonyms: Dict[str, str] = { _clean_token(k): _clean_token(v) for k, v in (occ.get("synonyms", {}) or {}).items() }

        # Silhouettes (not used yet, but loaded for later rules)
        sil = raw.get("silhouettes", {}) or {}
        self.allowed_silhouettes: Set[str] = set(sil.get("allowed", []))
        self.silhouette_norm: Dict[str, str] = { _clean_token(k): _clean_token(v) for k, v in (sil.get("normalize", {}) or {}).items() }

        # Outfit rules
        self.rules = (raw.get("rules", {}) or {})

    @staticmethod
    def load(path: str | Path | None = None) -> "Taxonomy":
        """
        Load taxonomy.yaml from the given path, or from the package directory by default.
        """
        if path is None:
            path = Path(__file__).with_name("taxonomy.yaml")
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Taxonomy(data or {})

    # -----------------------
    # Category normalization
    # -----------------------
    def canonicalize_category(self, subtype_or_category: str) -> str | None:
        """
        Map a free-text subtype or category to one of the canonical categories.
        Returns None if we cannot confidently map it.
        """
        token = _clean_token(subtype_or_category)
        if not token:
            return None
        # direct match
        if token in self.canonical_categories:
            return token
        # lookup in subtype map
        if token in self.subtype_to_category:
            return self.subtype_to_category[token]
        # heuristic fallbacks
        for key, cat in self.subtype_to_category.items():
            if key in token:
                return cat
        return None

    # -----------------------
    # Color normalization
    # -----------------------
    def normalize_colors(self, values: Iterable[str]) -> List[str]:
        out: List[str] = []
        for raw in values or []:
            t = _clean_token(str(raw))
            if not t:
                continue
            # split multi-token like "black/white"
            for tok in _split_tokens(t):
                # map synonyms
                tok = self.color_synonyms.get(tok, tok)
                if tok in self.palette:
                    out.append(tok)
        # dedupe but keep order
        seen = set()
        keep = []
        for c in out:
            if c not in seen:
                keep.append(c); seen.add(c)
        return keep[: self.max_dominant] if keep else []

    # -----------------------
    # Pattern normalization
    # -----------------------
    def normalize_patterns(self, values: Iterable[str]) -> List[str]:
        out: Set[str] = set()
        for raw in values or []:
            for tok in _split_tokens(raw):
                tok = self.pattern_norm.get(tok, tok)
                if tok in self.allowed_patterns:
                    out.add(tok)
                # common hint: if someone wrote "striped", we already map to "stripe" above
                elif tok.endswith("s") and tok[:-1] in self.allowed_patterns:
                    out.add(tok[:-1])
        return list(out)

    # -----------------------
    # Occasion normalization
    # -----------------------
    def normalize_occasions(self, values: Iterable[str]) -> List[str]:
        out: Set[str] = set()
        for raw in values or []:
            for tok in _split_tokens(raw):
                tok = self.occasion_synonyms.get(tok, tok)
                if tok in self.allowed_occasions:
                    out.add(tok)
        return list(out)


if __name__ == "__main__":
    # quick smoke test
    tx = Taxonomy.load()
    assert tx.canonicalize_category("Shirts & Blouses") == "top"
    assert tx.canonicalize_category("Denim") == "bottom"
    assert tx.normalize_colors(["Off-White", "black/white", "cognac"]) in (["ivory", "black", "white", "brown"][:tx.max_dominant], ["black","white","ivory","brown"][:tx.max_dominant])
    assert "stripe" in tx.normalize_patterns(["striped", "floral"])
    assert set(tx.normalize_occasions(["business casual", "night out"])) == {"work", "dinner"}
    print("taxonomy OK")
