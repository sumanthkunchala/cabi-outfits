# src/cabi_outfits/schema.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


_SPLIT = re.compile(r"[,/|]")


class Item(BaseModel):
    """
    Canonical product record after normalization.
    """
    product_id: str
    name: str
    season: Optional[str] = None

    # Canonical category (top|bottom|dress|outerwear|accessory) + original subtype
    category: Optional[str] = None
    subtype: Optional[str] = None

    price: Optional[float] = None
    image_url: Optional[str] = None

    colors: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    occasions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # ---------- helpers ----------
    @field_validator("colors", "patterns", "occasions", "tags", mode="before")
    @classmethod
    def _to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()]
        parts = _SPLIT.split(str(v))
        return [p.strip().lower() for p in parts if p.strip()]

    def key(self) -> str:
        return self.product_id

    def to_dict(self) -> dict:
        return self.model_dump()
