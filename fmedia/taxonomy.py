"""
taxonomy.py
-----------
Loads the monitoring taxonomy and the LLM prompts from a JSON file so that the
engine contains no topic-specific content. Default: examples/taxonomy.example.json
(invented, neutral categories). Point OMW_TAXONOMY at your own file to use your
own categories and prompts.

NOTE: database columns and a few identifiers keep their historical names
(`event_type`, `POSITIVE_TYPE_IDS`, ...) because this code was extracted from a
topic-specific system. They hold whatever categories the loaded taxonomy defines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, FrozenSet

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = _ROOT / "examples" / "taxonomy.example.json"


def _load() -> dict:
    path = Path(os.getenv("OMW_TAXONOMY", str(_DEFAULT)))
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["_dir"] = str(path.resolve().parent)
    return data


_TAX = _load()

NEGATIVE_TYPE: str = _TAX["negative"]["id"]
POSITIVE_TYPES: FrozenSet[str] = frozenset(t["id"] for t in _TAX["types"])
NON_FATAL_TYPES: FrozenSet[str] = frozenset(
    t["id"] for t in _TAX["types"] if t.get("non_fatal")
)
TYPE_LABELS: Dict[str, str] = {t["id"]: t["label"] for t in _TAX["types"]}
TYPE_LABELS[NEGATIVE_TYPE] = _TAX["negative"]["label"]
TYPE_LABELS_EN: Dict[str, str] = {t["id"]: t.get("label_en", t["label"]) for t in _TAX["types"]}
TYPE_LABELS_EN[NEGATIVE_TYPE] = _TAX["negative"].get("label_en", _TAX["negative"]["label"])
TYPE_COLORS: Dict[str, str] = {t["id"]: t.get("color", "#888888") for t in _TAX["types"]}
TYPE_COLORS[NEGATIVE_TYPE] = _TAX["negative"].get("color", "#6b7280")
TYPE_EMOJI: Dict[str, str] = {t["id"]: t.get("emoji", "•") for t in _TAX["types"]}

RELACIONES: FrozenSet[str] = frozenset(v["id"] for v in _TAX["relacion"]["values"])
RELACION_LABELS: Dict[str, str] = {v["id"]: v["label"] for v in _TAX["relacion"]["values"]}
RELACION_DEFAULT: str = _TAX["relacion"]["default"]
RELACION_NA: str = _TAX["relacion"]["not_applicable"]

MOTIVOS: FrozenSet[str] = frozenset(v["id"] for v in _TAX["motivo"]["values"])
MOTIVO_LABELS: Dict[str, str] = {v["id"]: v["label"] for v in _TAX["motivo"]["values"]}
MOTIVO_DEFAULT: str = _TAX["motivo"]["default"]
MOTIVO_NA: str = _TAX["motivo"]["not_applicable"]


def _read_prompt(key: str) -> str:
    return (Path(_TAX["_dir"]) / _TAX["prompts"][key]).read_text(encoding="utf-8")


# Optional: regexes that must ALL match (title+snippet) for a rejected article to be
# re-checked against its full text before the rejection becomes final. Empty = disabled.
import re as _re
RECHECK_REGEXES = [_re.compile(p, _re.IGNORECASE) for p in _TAX.get("recheck_regexes", [])]

CLASSIFIER_PROMPT: str = _read_prompt("classifier")
TRIAGE_PROMPT: str = _read_prompt("triage")
