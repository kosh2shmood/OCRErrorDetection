#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect likely OCR/field-pairing errors in Soviet patent certificate JSON exports.

The script is deliberately non-mutating: it only reads the source JSON files and
writes separate review reports. It expects Google Document AI-style JSON where
top-level entities have type "Entry" and nested properties named:

  certificate_number
  certificate_implementation
  certificate_value

Run from the export directory, for example:

  python3 scripts/detect_ocr_errors.py --input . --output output/ocr_error_reports
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FIELD_NUMBER = "certificate_number"
FIELD_IMPLEMENTATION = "certificate_implementation"
FIELD_VALUE = "certificate_value"
FIELD_TYPES = (FIELD_NUMBER, FIELD_IMPLEMENTATION, FIELD_VALUE)

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
SKIP_DIRS = {".git", "tmp", "output", "scripts", "docs", "__pycache__"}

FUZZY_PHRASE_THRESHOLD = 0.78
IMPLEMENTATION_PREFIX = "изобретение внедрено"
VALUE_PREFIX = "экономия от внедрения"
VALUE_SUFFIX = "в год"
PATENT_MARKER_RE = re.compile(
    r"(\(\s*(?:11|21|22|51|54|61|71|72)\s*\)|\bбюл\.?\b|авторск|патент)",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
NUMBER_ALLOWED_RE = re.compile(r"^\d+$")
VI_VOLUME_RE = re.compile(r"(?:ви\s*-\s*ii\s*)?(19\d{2}|20\d{2})\s*v\s*(\d+)", re.IGNORECASE)
COMPACT_VOLUME_RE = re.compile(r"^(19\d{2}|20\d{2})\s*[-_]\s*(\d+)", re.IGNORECASE)

LOCATION_CUE_RE = re.compile(
    r"(\bна\b|\bв\b|\bво\b|\bпри\b|\bпо\b|"
    r"завод|комбинат|фабрик|институт|нии|нпо|предприят|"
    r"трест|управлен|объединен|объединён|министер|шахт|цех|"
    r"колхоз|совхоз|\bг\.|\bим\.|сср|асср|обл\.?|край|район)",
    re.IGNORECASE,
)


@dataclass
class Box:
    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def page_number(self) -> int:
        return self.page_index + 1

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "x0": round(self.x0, 6),
            "y0": round(self.y0, 6),
            "x1": round(self.x1, 6),
            "y1": round(self.y1, 6),
            "cx": round(self.cx, 6),
            "cy": round(self.cy, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
        }


@dataclass
class Field:
    field_type: str
    text: str
    confidence: Optional[float]
    box: Optional[Box]
    id: Optional[str]

    @property
    def present(self) -> bool:
        return is_present_text(self.text) and self.box is not None


@dataclass
class Entry:
    json_path: str
    entry_index: int
    id: Optional[str]
    mention_text: str
    confidence: Optional[float]
    entry_box: Optional[Box]
    fields: Dict[str, Field]

    def field(self, field_type: str) -> Optional[Field]:
        field = self.fields.get(field_type)
        if field and field.present:
            return field
        return None

    @property
    def number(self) -> Optional[Field]:
        return self.field(FIELD_NUMBER)

    @property
    def implementation(self) -> Optional[Field]:
        return self.field(FIELD_IMPLEMENTATION)

    @property
    def value(self) -> Optional[Field]:
        return self.field(FIELD_VALUE)

    def primary_box(self) -> Optional[Box]:
        for field_type in (FIELD_NUMBER, FIELD_IMPLEMENTATION, FIELD_VALUE):
            field = self.field(field_type)
            if field and field.box:
                return field.box
        return self.entry_box

    def combined_field_box(self, page_index: Optional[int] = None) -> Optional[Box]:
        boxes = []
        for field_type in FIELD_TYPES:
            field = self.field(field_type)
            if field and field.box and (page_index is None or field.box.page_index == page_index):
                boxes.append(field.box)
        return combine_boxes(boxes)


@dataclass
class PageModel:
    page_index: int
    split_x: float
    split_confidence: float
    body_y0: float
    body_y1: float
    median_field_width: float
    field_count: int

    @property
    def body_height(self) -> float:
        return max(0.0, self.body_y1 - self.body_y0)


@dataclass
class RowItem:
    entry: Entry
    page_index: int
    column: int
    start_y: float
    end_y: float
    bbox: Box


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: Optional[str]) -> str:
    text = clean_text(text).lower().replace("ё", "е")
    return text


def normalize_letters(text: str) -> str:
    return re.sub(r"[^a-zа-я]+", "", normalize_text(text))


def normalize_certificate_number(text: Optional[str]) -> str:
    compact = re.sub(r"\s+", "", clean_text(text))
    if not NUMBER_ALLOWED_RE.fullmatch(compact):
        return ""
    return compact


def volume_key_from_path(path: str) -> str:
    """Return a stable same-volume key from known export filename patterns."""
    stem = Path(path).stem.lower().replace("ё", "е")
    stem = re.sub(r"\s+", " ", stem).strip()
    match = VI_VOLUME_RE.search(stem)
    if not match:
        match = COMPACT_VOLUME_RE.search(stem)
    if match:
        year, volume = match.groups()
        return f"{year}-v{int(volume)}"
    return stem


def is_present_text(text: Optional[str]) -> bool:
    text = clean_text(text)
    if not text:
        return False
    return text not in {"-", "--", "—", "–", "None", "null"}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quantile(values: Sequence[float], q: float, default: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[int(position)]
    return values[low] * (high - position) + values[high] * (position - low)


def robust_stats(values: Sequence[float]) -> Dict[str, float]:
    values = [v for v in values if math.isfinite(v) and v >= 0]
    if not values:
        return {"median": 0.0, "mad": 0.0, "p75": 0.0, "p90": 0.0}
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values) if len(values) > 1 else 0.0
    return {
        "median": median,
        "mad": mad,
        "p75": quantile(values, 0.75, median),
        "p90": quantile(values, 0.90, median),
    }


def box_from_page_ref(ref: Dict[str, Any], pages: Sequence[Dict[str, Any]]) -> Optional[Box]:
    poly = ref.get("boundingPoly") or {}
    vertices = poly.get("normalizedVertices")
    page_index = int(ref.get("page", 0) or 0)

    if vertices:
        xs = [to_float(v.get("x"), 0.0) for v in vertices]
        ys = [to_float(v.get("y"), 0.0) for v in vertices]
    else:
        vertices = poly.get("vertices")
        if not vertices:
            return None
        width = 1.0
        height = 1.0
        if 0 <= page_index < len(pages):
            dim = pages[page_index].get("dimension") or {}
            width = max(1.0, to_float(dim.get("width"), 1.0))
            height = max(1.0, to_float(dim.get("height"), 1.0))
        xs = [to_float(v.get("x"), 0.0) / width for v in vertices]
        ys = [to_float(v.get("y"), 0.0) / height for v in vertices]

    if not xs or not ys:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    if x1 < x0 or y1 < y0:
        return None
    return Box(page_index=page_index, x0=x0, y0=y0, x1=x1, y1=y1)


def box_from_entity(obj: Dict[str, Any], pages: Sequence[Dict[str, Any]]) -> Optional[Box]:
    refs = ((obj.get("pageAnchor") or {}).get("pageRefs") or [])
    boxes = [box_from_page_ref(ref, pages) for ref in refs]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return combine_boxes(boxes)


def combine_boxes(boxes: Sequence[Box]) -> Optional[Box]:
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    page_index = boxes[0].page_index
    same_page = [box for box in boxes if box.page_index == page_index]
    return Box(
        page_index=page_index,
        x0=min(box.x0 for box in same_page),
        y0=min(box.y0 for box in same_page),
        x1=max(box.x1 for box in same_page),
        y1=max(box.y1 for box in same_page),
    )


def field_from_property(prop: Dict[str, Any], pages: Sequence[Dict[str, Any]]) -> Field:
    return Field(
        field_type=prop.get("type", ""),
        text=prop.get("mentionText") or (prop.get("textAnchor") or {}).get("content") or "",
        confidence=prop.get("confidence"),
        box=box_from_entity(prop, pages),
        id=prop.get("id"),
    )


def entry_from_entity(
    json_path: str,
    entry_index: int,
    entity: Dict[str, Any],
    pages: Sequence[Dict[str, Any]],
) -> Entry:
    fields_by_type: Dict[str, List[Field]] = defaultdict(list)
    for prop in entity.get("properties") or []:
        field = field_from_property(prop, pages)
        if field.field_type in FIELD_TYPES:
            fields_by_type[field.field_type].append(field)

    fields = {}
    for field_type, candidates in fields_by_type.items():
        # Duplicate subfields are very rare. Prefer present fields, then confidence.
        candidates.sort(key=lambda f: (f.present, f.confidence if f.confidence is not None else -1), reverse=True)
        fields[field_type] = candidates[0]

    return Entry(
        json_path=json_path,
        entry_index=entry_index,
        id=entity.get("id"),
        mention_text=entity.get("mentionText") or (entity.get("textAnchor") or {}).get("content") or "",
        confidence=entity.get("confidence"),
        entry_box=box_from_entity(entity, pages),
        fields=fields,
    )


def dedupe_text(text: str) -> str:
    """Normalize only superficial spacing for exact duplicate comparison."""
    return clean_text(text).replace("ё", "е")


def dedupe_page_index(entry: Entry) -> Optional[int]:
    primary = entry.primary_box()
    if primary:
        return primary.page_index
    if entry.entry_box:
        return entry.entry_box.page_index
    return None


def dedupe_key(entry: Entry) -> Optional[Tuple[Any, ...]]:
    page_index = dedupe_page_index(entry)
    field_parts = []
    for field_type in FIELD_TYPES:
        field = entry.field(field_type)
        if field:
            field_parts.append((field_type, dedupe_text(field.text)))
    if not field_parts:
        return None
    return (entry.json_path, page_index, tuple(field_parts))


def remove_exact_duplicate_entries(entries: Sequence[Entry]) -> Tuple[List[Entry], Dict[str, Any]]:
    """Return entries with exact same-page text duplicates removed in memory.

    Duplicates are scoped to one JSON file/volume and page. Two entries are
    duplicates only when they have the same set of present subfields and the
    cleaned text of every present subfield is identical.
    """
    seen: Dict[Tuple[Any, ...], Entry] = {}
    kept: List[Entry] = []
    duplicate_examples = []
    duplicates_by_page: Counter[str] = Counter()

    for entry in entries:
        key = dedupe_key(entry)
        if key is None or key not in seen:
            if key is not None:
                seen[key] = entry
            kept.append(entry)
            continue

        original = seen[key]
        page_index = dedupe_page_index(entry)
        page_number = page_index + 1 if page_index is not None else None
        duplicates_by_page[str(page_number)] += 1
        if len(duplicate_examples) < 20:
            duplicate_examples.append(
                {
                    "page_number": page_number,
                    "kept_entry_index": original.entry_index,
                    "removed_entry_index": entry.entry_index,
                    "kept_entry_id": original.id,
                    "removed_entry_id": entry.id,
                    "certificate_number": clean_text(entry.number.text) if entry.number else "",
                    "certificate_implementation": text_excerpt(entry.implementation.text, 120)
                    if entry.implementation
                    else "",
                    "certificate_value": text_excerpt(entry.value.text, 120) if entry.value else "",
                }
            )

    stats = {
        "duplicates_removed": len(entries) - len(kept),
        "deduplicated_entry_count": len(kept),
        "duplicates_by_page": dict(sorted(duplicates_by_page.items(), key=lambda item: int(item[0]) if item[0] != "None" else -1)),
        "duplicate_examples": duplicate_examples,
    }
    return kept, stats


def is_standalone_number_entry(entry: Entry) -> bool:
    return entry.number is not None and entry.implementation is None and entry.value is None


def build_volume_reference_numbers(
    input_dir: Path,
    paths: Sequence[Path],
) -> Tuple[Dict[str, set], Dict[str, Any]]:
    """Collect real certificate numbers per same-volume group.

    A number is considered a reference when it appears in an entry that also has
    implementation or value text. Number-only entries never seed the reference
    set.
    """
    reference_numbers: Dict[str, set] = defaultdict(set)
    files_by_volume: Counter[str] = Counter()
    skipped_files = []

    for path in paths:
        rel_path = str(path.relative_to(input_dir))
        volume_key = volume_key_from_path(rel_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            skipped_files.append(
                {
                    "json_path": rel_path,
                    "volume_key": volume_key,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        files_by_volume[volume_key] += 1
        for index, entity in enumerate(data.get("entities") or []):
            if entity.get("type") != "Entry":
                continue
            field_texts: Dict[str, List[str]] = defaultdict(list)
            for prop in entity.get("properties") or []:
                field_type = prop.get("type")
                if field_type not in FIELD_TYPES:
                    continue
                text = prop.get("mentionText") or (prop.get("textAnchor") or {}).get("content") or ""
                if is_present_text(text):
                    field_texts[field_type].append(text)
            if not field_texts.get(FIELD_NUMBER) or not (
                field_texts.get(FIELD_IMPLEMENTATION) or field_texts.get(FIELD_VALUE)
            ):
                continue
            for number_text in field_texts[FIELD_NUMBER]:
                normalized_number = normalize_certificate_number(number_text)
                if normalized_number:
                    reference_numbers[volume_key].add(normalized_number)

    stats = {
        "reference_files_scanned": len(paths),
        "reference_files_failed": len(skipped_files),
        "reference_file_failures": skipped_files[:20],
        "volume_count": len(reference_numbers),
        "reference_numbers_by_volume": {
            volume: len(numbers) for volume, numbers in sorted(reference_numbers.items())
        },
        "reference_files_by_volume": dict(sorted(files_by_volume.items())),
    }
    return reference_numbers, stats


def remove_standalone_index_numbers(
    entries: Sequence[Entry],
    volume_key: str,
    reference_numbers: Dict[str, set],
) -> Tuple[List[Entry], Dict[str, Any]]:
    numbers_for_volume = reference_numbers.get(volume_key, set())
    if not numbers_for_volume:
        return entries if isinstance(entries, list) else list(entries), {
            "standalone_index_numbers_ignored": 0,
            "ignored_index_numbers_by_page": {},
            "ignored_index_number_examples": [],
        }

    kept = []
    ignored_by_page: Counter[str] = Counter()
    examples = []

    for entry in entries:
        number = entry.number
        normalized_number = normalize_certificate_number(number.text if number else "")
        should_ignore = (
            is_standalone_number_entry(entry)
            and normalized_number
            and normalized_number in numbers_for_volume
        )
        if not should_ignore:
            kept.append(entry)
            continue

        page_index = dedupe_page_index(entry)
        page_number = page_index + 1 if page_index is not None else None
        ignored_by_page[str(page_number)] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "page_number": page_number,
                    "entry_index": entry.entry_index,
                    "entry_id": entry.id,
                    "certificate_number": normalized_number,
                    "volume_key": volume_key,
                }
            )

    stats = {
        "standalone_index_numbers_ignored": len(entries) - len(kept),
        "ignored_index_numbers_by_page": dict(
            sorted(ignored_by_page.items(), key=lambda item: int(item[0]) if item[0] != "None" else -1)
        ),
        "ignored_index_number_examples": examples,
    }
    return kept, stats


def infer_page_models(entries: Sequence[Entry], page_count: int) -> Dict[int, PageModel]:
    field_boxes_by_page: Dict[int, List[Tuple[str, Box]]] = defaultdict(list)
    for entry in entries:
        for field_type in FIELD_TYPES:
            field = entry.field(field_type)
            if field and field.box:
                field_boxes_by_page[field.box.page_index].append((field_type, field.box))

    models = {}
    for page_index in range(page_count):
        typed_boxes = field_boxes_by_page.get(page_index, [])
        boxes = [box for _, box in typed_boxes]
        usable = [box for box in boxes if 0.005 <= box.width <= 0.78]
        number_signal = [
            box
            for field_type, box in typed_boxes
            if field_type == FIELD_NUMBER and 0.005 <= box.width <= 0.18
        ]
        # Narrow number boxes are the best column signal. Wide implementation
        # boxes often reach close to the gutter and can make the split land
        # inside a true column if used too early.
        column_signal = number_signal if len(number_signal) >= 4 else usable
        centers = sorted(box.cx for box in column_signal if 0.01 <= box.cx <= 0.99)
        split_x = 0.5
        split_confidence = 0.0

        if len(centers) >= 6:
            gaps = []
            for i in range(len(centers) - 1):
                left_count = i + 1
                right_count = len(centers) - left_count
                if left_count < 2 or right_count < 2:
                    continue
                gap = centers[i + 1] - centers[i]
                gaps.append((gap, i))
            if gaps:
                gap, i = max(gaps, key=lambda item: item[0])
                median_gap = statistics.median([g for g, _ in gaps]) if gaps else 0.0
                if gap >= max(0.08, median_gap * 2.5):
                    split_x = (centers[i] + centers[i + 1]) / 2.0
                    split_confidence = min(1.0, gap / 0.25)

        if split_confidence > 0:
            left_number_edges = [box.x1 for box in number_signal if box.cx < split_x]
            right_number_edges = [box.x0 for box in number_signal if box.cx >= split_x]
            left_text_edges = [
                box.x1
                for field_type, box in typed_boxes
                if field_type in {FIELD_IMPLEMENTATION, FIELD_VALUE}
                and 0.05 <= box.width <= 0.78
                and box.cx < split_x
            ]
            right_text_edges = [
                box.x0
                for field_type, box in typed_boxes
                if field_type in {FIELD_IMPLEMENTATION, FIELD_VALUE}
                and 0.05 <= box.width <= 0.78
                and box.cx >= split_x
            ]
            if len(left_text_edges) >= 2 and len(right_text_edges) >= 2:
                left_edge = quantile(left_text_edges, 0.80, max(left_text_edges))
                right_edge = quantile(right_text_edges, 0.20, min(right_text_edges))
                if right_edge > left_edge + 0.01:
                    split_x = (left_edge + right_edge) / 2.0
            elif len(left_text_edges) >= 2 and len(right_number_edges) >= 2:
                left_edge = quantile(left_text_edges, 0.80, max(left_text_edges))
                right_edge = quantile(right_number_edges, 0.20, min(right_number_edges))
                if right_edge > left_edge + 0.01:
                    split_x = (left_edge + right_edge) / 2.0
            elif len(right_text_edges) >= 2 and len(left_number_edges) >= 2:
                left_edge = quantile(left_number_edges, 0.80, max(left_number_edges))
                right_edge = quantile(right_text_edges, 0.20, min(right_text_edges))
                if right_edge > left_edge + 0.01:
                    split_x = (left_edge + right_edge) / 2.0

        y0s = [box.y0 for box in usable]
        y1s = [box.y1 for box in usable]
        text_widths = [
            box.width
            for field_type, box in typed_boxes
            if field_type in {FIELD_IMPLEMENTATION, FIELD_VALUE} and 0.005 < box.width <= 0.78
        ]
        widths = text_widths or [box.width for box in usable if box.width > 0.005]
        models[page_index] = PageModel(
            page_index=page_index,
            split_x=split_x,
            split_confidence=split_confidence,
            body_y0=quantile(y0s, 0.05, 0.05),
            body_y1=quantile(y1s, 0.95, 0.95),
            median_field_width=statistics.median(widths) if widths else 0.25,
            field_count=len(boxes),
        )
    return models


def column_for_box(box: Box, model: PageModel) -> int:
    return 0 if box.cx < model.split_x else 1


def top_band(model: PageModel) -> float:
    if model.body_height <= 0:
        return 0.25
    return min(0.45, model.body_y0 + model.body_height * 0.30)


def bottom_band(model: PageModel) -> float:
    if model.body_height <= 0:
        return 0.78
    return max(0.72, model.body_y1 - model.body_height * 0.25)


def is_top_left(box: Box, model: PageModel) -> bool:
    return column_for_box(box, model) == 0 and box.cy <= top_band(model)


def is_top_right(box: Box, model: PageModel) -> bool:
    return column_for_box(box, model) == 1 and box.cy <= top_band(model)


def touches_right_column(box: Box, model: PageModel, margin: float = 0.04) -> bool:
    return box.x1 >= model.split_x + margin


def starts_in_top_right(box: Box, model: PageModel) -> bool:
    # For valid column wraps, the implementation may be overboxed and span a
    # large part of the page. Use its top edge and right-column overlap rather
    # than its center point, which can be dragged toward the middle/left.
    return box.y0 <= top_band(model) and touches_right_column(box, model)


def is_bottom_left(box: Box, model: PageModel) -> bool:
    return column_for_box(box, model) == 0 and box.cy >= bottom_band(model)


def is_bottom_right(box: Box, model: PageModel) -> bool:
    return column_for_box(box, model) == 1 and box.cy >= bottom_band(model)


def allowed_same_page_column_wrap(number_box: Box, implementation_box: Box, models: Dict[int, PageModel]) -> bool:
    model = models.get(number_box.page_index)
    if model is None or number_box.page_index != implementation_box.page_index:
        return False
    return is_bottom_left(number_box, model) and starts_in_top_right(implementation_box, model)


def field_spans_split(box: Box, model: PageModel, margin: float = 0.02) -> bool:
    return box.x0 < model.split_x - margin and box.x1 > model.split_x + margin


def phrase_start_similarity(text: str, phrase: str) -> float:
    target = normalize_letters(phrase)
    letters = normalize_letters(text)
    if not target or not letters:
        return 0.0
    return difflib.SequenceMatcher(None, letters[: len(target)], target).ratio()


def phrase_end_similarity(text: str, phrase: str) -> float:
    target = normalize_letters(phrase)
    letters = normalize_letters(text)
    if not target or not letters:
        return 0.0
    return difflib.SequenceMatcher(None, letters[-len(target) :], target).ratio()


def starts_like_phrase(text: str, phrase: str) -> bool:
    norm = normalize_text(text)
    if norm.startswith(phrase):
        return True
    return phrase_start_similarity(text, phrase) >= FUZZY_PHRASE_THRESHOLD


def ends_like_phrase(text: str, phrase: str) -> bool:
    norm = normalize_text(text).rstrip(" .,:;!?)]}\"'»”")
    if norm.endswith(phrase):
        return True
    return phrase_end_similarity(text, phrase) >= FUZZY_PHRASE_THRESHOLD


def starts_like_implementation(text: str) -> bool:
    return starts_like_phrase(text, IMPLEMENTATION_PREFIX)


def implementation_lacks_location_after_year(text: str, min_chars: int) -> Tuple[bool, Dict[str, Any]]:
    impl_norm = normalize_text(text)
    year_match = YEAR_RE.search(impl_norm)
    if not starts_like_implementation(text) or not year_match:
        return False, {
            "normalized_length": len(impl_norm),
            "has_year": bool(year_match),
            "has_location_cue_after_year": None,
        }
    after_year = impl_norm[year_match.end() :]
    has_location_cue = bool(LOCATION_CUE_RE.search(after_year))
    suspicious = len(impl_norm) < min_chars or not has_location_cue
    return suspicious, {
        "normalized_length": len(impl_norm),
        "has_year": True,
        "has_location_cue_after_year": has_location_cue,
    }


def text_excerpt(text: str, max_chars: int = 160) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def issue_context(entry: Entry, models: Dict[int, PageModel]) -> Dict[str, Any]:
    primary = entry.primary_box()
    page_index = primary.page_index if primary else None
    model = models.get(page_index) if page_index is not None else None
    field_bboxes = {}
    field_columns = {}
    field_confidence = {}
    for field_type in FIELD_TYPES:
        field = entry.field(field_type)
        field_bboxes[field_type] = field.box.as_dict() if field and field.box else None
        field_confidence[field_type] = field.confidence if field else None
        if field and field.box and field.box.page_index in models:
            field_columns[field_type] = column_for_box(field.box, models[field.box.page_index])
        else:
            field_columns[field_type] = None

    return {
        "json_path": entry.json_path,
        "entry_index": entry.entry_index,
        "entry_id": entry.id,
        "page_index": page_index,
        "page_number": page_index + 1 if page_index is not None else None,
        "column": column_for_box(primary, model) if primary and model else None,
        "entry_bbox": entry.entry_box.as_dict() if entry.entry_box else None,
        "field_bboxes": field_bboxes,
        "field_columns": field_columns,
        "field_confidence": field_confidence,
        "certificate_number": clean_text(entry.number.text) if entry.number else "",
        "certificate_implementation": text_excerpt(entry.implementation.text, 240) if entry.implementation else "",
        "certificate_value": text_excerpt(entry.value.text, 240) if entry.value else "",
    }


def make_issue(
    code: str,
    severity: str,
    entry: Entry,
    message: str,
    models: Dict[int, PageModel],
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = issue_context(entry, models)
    record.update(
        {
            "issue_code": code,
            "severity": severity,
            "message": message,
            "evidence": evidence or {},
        }
    )
    return record


def continuation_pairs(entries: Sequence[Entry], models: Dict[int, PageModel]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    number_only_by_page: Dict[int, List[Entry]] = defaultdict(list)
    implementation_only_by_page: Dict[int, List[Entry]] = defaultdict(list)
    for entry in entries:
        number = entry.number
        implementation = entry.implementation
        if number and not implementation and number.box:
            model = models.get(number.box.page_index)
            if model and is_bottom_right(number.box, model):
                number_only_by_page[number.box.page_index].append(entry)
        if implementation and not number and implementation.box:
            model = models.get(implementation.box.page_index)
            if model and is_top_left(implementation.box, model):
                implementation_only_by_page[implementation.box.page_index].append(entry)

    paired: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for page_index, numbers in number_only_by_page.items():
        implementations = implementation_only_by_page.get(page_index + 1, [])
        if not implementations:
            continue
        numbers = sorted(numbers, key=lambda e: e.number.box.cy if e.number and e.number.box else 0, reverse=True)
        implementations = sorted(
            implementations,
            key=lambda e: e.implementation.box.cy if e.implementation and e.implementation.box else 1,
        )
        for number_entry, implementation_entry in zip(numbers, implementations):
            pair = {
                "paired_exception": "bottom_right_number_to_next_page_top_left_implementation",
                "number_entry_index": number_entry.entry_index,
                "implementation_entry_index": implementation_entry.entry_index,
                "number_page_number": page_index + 1,
                "implementation_page_number": page_index + 2,
            }
            paired[(number_entry.json_path, number_entry.entry_index)] = pair
            paired[(implementation_entry.json_path, implementation_entry.entry_index)] = pair
    return paired


def build_row_items(entries: Sequence[Entry], models: Dict[int, PageModel]) -> List[RowItem]:
    rows = []
    for entry in entries:
        primary = entry.primary_box()
        if primary is None:
            continue
        model = models.get(primary.page_index)
        if model is None:
            continue
        bbox = entry.combined_field_box(primary.page_index) or entry.entry_box or primary
        rows.append(
            RowItem(
                entry=entry,
                page_index=primary.page_index,
                column=column_for_box(primary, model),
                start_y=primary.y0,
                end_y=bbox.y1,
                bbox=bbox,
            )
        )
    return rows


def spacing_issues(entries: Sequence[Entry], models: Dict[int, PageModel], args: argparse.Namespace) -> List[Dict[str, Any]]:
    issues = []
    rows_by_group: Dict[Tuple[int, int], List[RowItem]] = defaultdict(list)
    for row in build_row_items(entries, models):
        # Very wide rows are already suspicious and should not define the normal spacing baseline.
        model = models.get(row.page_index)
        if model and not field_spans_split(row.bbox, model, margin=0.03):
            rows_by_group[(row.page_index, row.column)].append(row)

    for (page_index, column), rows in rows_by_group.items():
        rows.sort(key=lambda row: (row.start_y, row.bbox.x0))
        if len(rows) < 4:
            continue
        start_deltas = [
            rows[i + 1].start_y - rows[i].start_y
            for i in range(len(rows) - 1)
            if rows[i + 1].start_y > rows[i].start_y
        ]
        whitespace_gaps = [
            rows[i + 1].bbox.y0 - rows[i].bbox.y1
            for i in range(len(rows) - 1)
            if rows[i + 1].bbox.y0 > rows[i].bbox.y1
        ]
        start_stats = robust_stats(start_deltas)
        gap_stats = robust_stats(whitespace_gaps)
        start_threshold = max(
            args.min_large_start_delta,
            start_stats["median"] * args.start_delta_ratio,
            start_stats["median"] + (1.4826 * start_stats["mad"] * args.mad_multiplier),
            start_stats["p90"] * 1.15,
        )
        gap_threshold = max(
            args.min_large_whitespace_gap,
            gap_stats["median"] * args.whitespace_gap_ratio,
            gap_stats["median"] + (1.4826 * gap_stats["mad"] * args.mad_multiplier),
            gap_stats["p90"] * 1.15,
        )

        for i in range(len(rows) - 1):
            current = rows[i]
            nxt = rows[i + 1]
            start_delta = nxt.start_y - current.start_y
            whitespace_gap = nxt.bbox.y0 - current.bbox.y1
            if start_delta > start_threshold:
                issues.append(
                    make_issue(
                        "large_start_delta_after_entry",
                        "warning",
                        current.entry,
                        "Large vertical distance to the next detected entry in this page column; inspect for a missing whole entry or a truncated/missing implementation/value.",
                        models,
                        {
                            "page_number": page_index + 1,
                            "column": column,
                            "next_entry_index": nxt.entry.entry_index,
                            "start_delta": round(start_delta, 6),
                            "threshold": round(start_threshold, 6),
                            "median_start_delta": round(start_stats["median"], 6),
                        },
                    )
                )
            if whitespace_gap > gap_threshold and not current.entry.value:
                issues.append(
                    make_issue(
                        "possible_missing_value_or_trailing_text",
                        "warning",
                        current.entry,
                        "Large blank/unclaimed space after an entry without certificate_value; inspect for a missing value, missing location, or omitted trailing implementation text.",
                        models,
                        {
                            "page_number": page_index + 1,
                            "column": column,
                            "next_entry_index": nxt.entry.entry_index,
                            "whitespace_gap": round(whitespace_gap, 6),
                            "threshold": round(gap_threshold, 6),
                            "median_whitespace_gap": round(gap_stats["median"], 6),
                        },
                    )
                )
    return issues


def valid_pair_for_implementation_gap(row: RowItem, nxt: RowItem, models: Dict[int, PageModel]) -> bool:
    if not row.entry.number or not row.entry.implementation:
        return False
    if not nxt.entry.number or not nxt.entry.implementation:
        return False
    number = row.entry.number
    implementation = row.entry.implementation
    next_number = nxt.entry.number
    next_implementation = nxt.entry.implementation
    if not (number.box and implementation.box and next_number.box and next_implementation.box):
        return False
    if implementation.box.page_index != row.page_index or next_implementation.box.page_index != nxt.page_index:
        return False
    model = models.get(row.page_index)
    if model is None:
        return False
    if allowed_same_page_column_wrap(number.box, implementation.box, models):
        return False
    return column_for_box(number.box, model) == row.column and column_for_box(implementation.box, model) == row.column


def implementation_to_next_entry_gap_issues(
    entries: Sequence[Entry],
    models: Dict[int, PageModel],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    issues = []
    rows_by_group: Dict[Tuple[int, int], List[RowItem]] = defaultdict(list)
    for row in build_row_items(entries, models):
        rows_by_group[(row.page_index, row.column)].append(row)

    for (page_index, column), rows in rows_by_group.items():
        rows.sort(key=lambda row: (row.start_y, row.bbox.x0))
        if len(rows) < 4:
            continue

        preferred_gaps = []
        fallback_gaps = []
        for current, nxt in zip(rows, rows[1:]):
            implementation = current.entry.implementation
            if not implementation or not implementation.box:
                continue
            gap = nxt.start_y - implementation.box.y1
            if gap <= 0:
                continue
            if valid_pair_for_implementation_gap(current, nxt, models):
                fallback_gaps.append(gap)
                if current.entry.value:
                    preferred_gaps.append(gap)

        baseline_gaps = preferred_gaps if len(preferred_gaps) >= args.implementation_gap_min_baseline_pairs else fallback_gaps
        if len(baseline_gaps) < args.implementation_gap_min_baseline_pairs:
            continue

        stats = robust_stats(baseline_gaps)
        threshold = max(
            args.min_implementation_to_next_gap,
            stats["median"] * args.implementation_to_next_gap_ratio,
            stats["median"] + (1.4826 * stats["mad"] * args.mad_multiplier),
            stats["p90"] * 1.15,
        )
        baseline_source = "valid_number_implementation_with_value" if baseline_gaps is preferred_gaps else "valid_number_implementation_all"

        for current, nxt in zip(rows, rows[1:]):
            implementation = current.entry.implementation
            if not implementation or not implementation.box:
                continue
            number = current.entry.number
            if number and number.box and allowed_same_page_column_wrap(number.box, implementation.box, models):
                continue
            gap = nxt.start_y - implementation.box.y1
            if gap <= threshold:
                continue

            text_suspicious, text_evidence = implementation_lacks_location_after_year(
                implementation.text,
                args.min_implementation_chars,
            )
            if current.entry.value and not text_suspicious:
                continue

            issues.append(
                make_issue(
                    "implementation_maybe_truncated_no_location",
                    "warning",
                    current.entry,
                    "Large gap from the end of the implementation box to the next entry in the same page column; inspect for a truncated implementation or a missed certificate_value.",
                    models,
                    {
                        "page_number": page_index + 1,
                        "column": column,
                        "next_entry_index": nxt.entry.entry_index,
                        "implementation_to_next_gap": round(gap, 6),
                        "threshold": round(threshold, 6),
                        "median_baseline_gap": round(stats["median"], 6),
                        "p90_baseline_gap": round(stats["p90"], 6),
                        "baseline_pair_count": len(baseline_gaps),
                        "baseline_source": baseline_source,
                        "current_has_value": bool(current.entry.value),
                        "text_suspicious": text_suspicious,
                        **text_evidence,
                    },
                )
            )
    return issues


def implementation_value_gap_issues(
    entries: Sequence[Entry],
    models: Dict[int, PageModel],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Flag unusually large vertical gaps between implementation and value.

    A value normally follows soon after the implementation text. A large gap
    between the bottom of the implementation box and the top of the value box
    often means the implementation location/tail was missed.
    """
    candidates = []
    gaps_by_page_column: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    gaps_by_page: Dict[int, List[float]] = defaultdict(list)
    file_gaps: List[float] = []

    for entry in entries:
        implementation = entry.implementation
        value = entry.value
        if not implementation or not value or not implementation.box or not value.box:
            continue
        if implementation.box.page_index != value.box.page_index:
            continue

        page_index = implementation.box.page_index
        model = models.get(page_index)
        if model is None:
            continue
        implementation_col = column_for_box(implementation.box, model)
        value_col = column_for_box(value.box, model)
        if implementation_col != value_col:
            continue
        if field_spans_split(implementation.box, model, margin=0.03) or field_spans_split(
            value.box, model, margin=0.03
        ):
            continue

        gap = value.box.y0 - implementation.box.y1
        if gap <= 0:
            continue

        item = {
            "entry": entry,
            "page_index": page_index,
            "column": implementation_col,
            "gap": gap,
        }
        candidates.append(item)
        gaps_by_page_column[(page_index, implementation_col)].append(gap)
        gaps_by_page[page_index].append(gap)
        file_gaps.append(gap)

    if len(file_gaps) < args.implementation_value_gap_min_baseline_pairs:
        return []

    issues = []
    for item in candidates:
        page_index = item["page_index"]
        column = item["column"]
        gap = item["gap"]

        baseline_source = "same_page_column"
        baseline_gaps = gaps_by_page_column[(page_index, column)]
        if len(baseline_gaps) < args.implementation_value_gap_min_baseline_pairs:
            baseline_source = "same_page"
            baseline_gaps = gaps_by_page[page_index]
        if len(baseline_gaps) < args.implementation_value_gap_min_baseline_pairs:
            baseline_source = "same_file"
            baseline_gaps = file_gaps

        if len(baseline_gaps) < args.implementation_value_gap_min_baseline_pairs:
            continue

        stats = robust_stats(baseline_gaps)
        threshold = max(
            args.min_implementation_value_gap,
            stats["median"] * args.implementation_value_gap_ratio,
            stats["median"] + (1.4826 * stats["mad"] * args.mad_multiplier),
            stats["p90"] * 1.15,
        )
        if gap <= threshold:
            continue

        issues.append(
            make_issue(
                "implementation_value_large_gap",
                "warning",
                item["entry"],
                "Large vertical gap between certificate_implementation and certificate_value; inspect for missing implementation text between them.",
                models,
                {
                    "page_number": page_index + 1,
                    "column": column,
                    "implementation_value_gap": round(gap, 6),
                    "threshold": round(threshold, 6),
                    "median_baseline_gap": round(stats["median"], 6),
                    "p90_baseline_gap": round(stats["p90"], 6),
                    "baseline_pair_count": len(baseline_gaps),
                    "baseline_source": baseline_source,
                },
            )
        )

    return issues


def page_top_left_start_issues(
    entries: Sequence[Entry],
    models: Dict[int, PageModel],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows_by_page: Dict[int, List[RowItem]] = defaultdict(list)
    for row in build_row_items(entries, models):
        rows_by_page[row.page_index].append(row)

    issues = []
    for page_index, rows in rows_by_page.items():
        if len(rows) < args.first_entry_min_page_rows:
            continue
        model = models.get(page_index)
        if model is None:
            continue

        rows.sort(key=lambda row: (row.start_y, row.column, row.bbox.x0))
        left_rows = sorted(
            [row for row in rows if row.column == 0],
            key=lambda row: (row.start_y, row.bbox.x0),
        )
        earliest = rows[0]
        first_left = left_rows[0] if left_rows else None

        earliest_start = earliest.start_y
        first_left_start = first_left.start_y if first_left else None
        top_limit = max(args.first_entry_top_absolute_threshold, earliest_start + args.first_entry_top_relative_delta)

        left_column_missing_at_top = first_left is None or first_left.start_y > top_limit
        page_starts_too_low = earliest_start > args.first_entry_top_absolute_threshold
        if not left_column_missing_at_top and not page_starts_too_low:
            continue

        anchor_row = first_left or earliest
        issues.append(
            make_issue(
                "page_first_entry_not_top_left",
                "error",
                anchor_row.entry,
                "The page does not appear to have a detected first entry starting near the top-left; inspect for a missing whole entry or missing beginning of an implementation.",
                models,
                {
                    "page_number": page_index + 1,
                    "row_count_on_page": len(rows),
                    "earliest_entry_index": earliest.entry.entry_index,
                    "earliest_start_y": round(earliest_start, 6),
                    "first_left_entry_index": first_left.entry.entry_index if first_left else None,
                    "first_left_start_y": round(first_left_start, 6) if first_left_start is not None else None,
                    "top_limit": round(top_limit, 6),
                    "absolute_top_threshold": args.first_entry_top_absolute_threshold,
                    "relative_top_delta": args.first_entry_top_relative_delta,
                    "left_column_missing_at_top": left_column_missing_at_top,
                    "page_starts_too_low": page_starts_too_low,
                },
            )
        )
    return issues


def adjacent_missing_field_issues(
    entries: Sequence[Entry],
    models: Dict[int, PageModel],
    paired_exceptions: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_group: Dict[Tuple[int, int], List[RowItem]] = defaultdict(list)
    for row in build_row_items(entries, models):
        rows_by_group[(row.page_index, row.column)].append(row)

    issues = []
    for (page_index, column), rows in rows_by_group.items():
        rows.sort(key=lambda row: (row.start_y, row.bbox.x0))
        for left, right in zip(rows, rows[1:]):
            if (left.entry.json_path, left.entry.entry_index) in paired_exceptions:
                continue
            if (right.entry.json_path, right.entry.entry_index) in paired_exceptions:
                continue
            left_num_only = left.entry.number and not left.entry.implementation
            right_impl_only = right.entry.implementation and not right.entry.number
            left_impl_only = left.entry.implementation and not left.entry.number
            right_num_only = right.entry.number and not right.entry.implementation
            if (left_num_only and right_impl_only) or (left_impl_only and right_num_only):
                issues.append(
                    make_issue(
                        "adjacent_complementary_missing_fields",
                        "error",
                        left.entry,
                        "Adjacent entries have complementary missing fields; avoid automatically pairing them without manual review.",
                        models,
                        {
                            "page_number": page_index + 1,
                            "column": column,
                            "neighbor_entry_index": right.entry.entry_index,
                            "left_has_number": bool(left.entry.number),
                            "left_has_implementation": bool(left.entry.implementation),
                            "right_has_number": bool(right.entry.number),
                            "right_has_implementation": bool(right.entry.implementation),
                        },
                    )
                )
    return issues


def check_entry(
    entry: Entry,
    models: Dict[int, PageModel],
    paired_exceptions: Dict[Tuple[str, int], Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    issues = []
    number = entry.number
    implementation = entry.implementation
    value = entry.value
    pair_key = (entry.json_path, entry.entry_index)

    if not number:
        if pair_key in paired_exceptions:
            if args.emit_allowed_continuations:
                issues.append(
                    make_issue(
                        "allowed_cross_page_continuation",
                        "info",
                        entry,
                        "Implementation-only entry matches the allowed top-left continuation from a bottom-right number on the previous page.",
                        models,
                        paired_exceptions[pair_key],
                    )
                )
        else:
            issues.append(
                make_issue(
                    "missing_certificate_number",
                    "error",
                    entry,
                    "Entry has no certificate_number and does not match the allowed next-page continuation pattern.",
                    models,
                )
            )

    if not implementation:
        if pair_key in paired_exceptions:
            if args.emit_allowed_continuations:
                issues.append(
                    make_issue(
                        "allowed_cross_page_continuation",
                        "info",
                        entry,
                        "Number-only entry matches the allowed bottom-right continuation to a top-left implementation on the next page.",
                        models,
                        paired_exceptions[pair_key],
                    )
                )
        else:
            issues.append(
                make_issue(
                    "missing_certificate_implementation",
                    "error",
                    entry,
                    "Entry has no certificate_implementation and does not match the allowed bottom-right continuation pattern.",
                    models,
                )
            )

    if number:
        number_text = clean_text(number.text)
        compact = re.sub(r"\s+", "", number_text)
        if not NUMBER_ALLOWED_RE.fullmatch(compact):
            issues.append(
                make_issue(
                    "certificate_number_non_digit",
                    "error",
                    entry,
                    "Certificate number contains non-digit characters.",
                    models,
                    {"certificate_number_raw": number_text},
                )
            )
        if len(compact) < args.min_certificate_number_length or len(compact) > args.max_certificate_number_length:
            issues.append(
                make_issue(
                    "certificate_number_bad_length",
                    "error",
                    entry,
                    "Certificate number length is outside the expected 5-7 digit range.",
                    models,
                    {
                        "certificate_number_raw": number_text,
                        "length_after_space_removal": len(compact),
                        "min_length": args.min_certificate_number_length,
                        "max_length": args.max_certificate_number_length,
                    },
                )
            )

    if implementation:
        impl_text = implementation.text
        impl_norm = normalize_text(impl_text)
        if not starts_like_implementation(impl_text):
            issues.append(
                make_issue(
                    "implementation_bad_prefix",
                    "warning",
                    entry,
                    'Implementation does not appear to start with "Изобретение внедрено".',
                    models,
                    {"implementation_excerpt": text_excerpt(impl_text)},
                )
            )

        digit_count = sum(ch.isdigit() for ch in impl_text)
        alnum_count = sum(ch.isalnum() for ch in impl_text)
        digit_ratio = digit_count / alnum_count if alnum_count else 0.0
        marker_hits = PATENT_MARKER_RE.findall(impl_text)
        if digit_ratio >= args.implementation_digit_ratio:
            issues.append(
                make_issue(
                    "implementation_high_digit_ratio",
                    "warning",
                    entry,
                    "Implementation text has an unusually high digit ratio; inspect for captured patent metadata or spillover.",
                    models,
                    {
                        "digit_ratio": round(digit_ratio, 4),
                        "digit_count": digit_count,
                        "alnum_count": alnum_count,
                    },
                )
            )
        if marker_hits:
            issues.append(
                make_issue(
                    "implementation_contains_patent_markers",
                    "warning",
                    entry,
                    "Implementation text contains patent metadata markers such as (11), (21), (54), or bulletin text.",
                    models,
                    {"marker_count": len(marker_hits), "implementation_excerpt": text_excerpt(impl_text)},
                )
            )

        if implementation.confidence is not None and implementation.confidence < args.low_confidence_threshold:
            issues.append(
                make_issue(
                    "low_implementation_confidence",
                    "warning",
                    entry,
                    "Implementation field confidence is low.",
                    models,
                    {"confidence": implementation.confidence},
                )
            )

    if value:
        value_text = value.text
        value_prefix_ok = starts_like_phrase(value_text, VALUE_PREFIX)
        value_suffix_ok = ends_like_phrase(value_text, VALUE_SUFFIX)
        if not value_prefix_ok or not value_suffix_ok:
            issues.append(
                make_issue(
                    "value_bad_prefix",
                    "warning",
                    entry,
                    'Value does not appear to start with "Экономия от внедрения" and end with "в год".',
                    models,
                    {
                        "value_excerpt": text_excerpt(value_text),
                        "expected_prefix": VALUE_PREFIX,
                        "expected_suffix": VALUE_SUFFIX,
                        "prefix_ok": value_prefix_ok,
                        "suffix_ok": value_suffix_ok,
                        "prefix_similarity": round(phrase_start_similarity(value_text, VALUE_PREFIX), 4),
                        "suffix_similarity": round(phrase_end_similarity(value_text, VALUE_SUFFIX), 4),
                        "fuzzy_threshold": FUZZY_PHRASE_THRESHOLD,
                    },
                )
            )

    same_page_wrap_allowed = (
        number is not None
        and implementation is not None
        and number.box is not None
        and implementation.box is not None
        and number.box.page_index == implementation.box.page_index
        and allowed_same_page_column_wrap(number.box, implementation.box, models)
    )

    for field_type, field in ((FIELD_IMPLEMENTATION, implementation), (FIELD_VALUE, value)):
        if not field or not field.box:
            continue
        model = models.get(field.box.page_index)
        if not model:
            continue
        if field_type == FIELD_IMPLEMENTATION and same_page_wrap_allowed:
            continue
        span_width_threshold = max(0.55, model.median_field_width * 1.35)
        wide_width_threshold = max(0.60, model.median_field_width * args.wide_field_width_ratio)
        spans_problem = field_spans_split(field.box, model) and field.box.width >= span_width_threshold
        wide_problem = field.box.width >= wide_width_threshold
        if spans_problem or wide_problem:
            issues.append(
                make_issue(
                    "field_spans_multiple_columns",
                    "warning",
                    entry,
                    f"{field_type} is unusually wide or spans the inferred column split; inspect for text spillover or same-row merge.",
                    models,
                    {
                        "field_type": field_type,
                        "field_width": round(field.box.width, 6),
                        "page_split_x": round(model.split_x, 6),
                        "median_field_width": round(model.median_field_width, 6),
                        "span_width_threshold": round(span_width_threshold, 6),
                        "wide_width_threshold": round(wide_width_threshold, 6),
                    },
                )
            )

    if number and implementation and number.box and implementation.box:
        if number.box.page_index == implementation.box.page_index:
            model = models.get(number.box.page_index)
            if model:
                number_col = column_for_box(number.box, model)
                implementation_col = column_for_box(implementation.box, model)
                allowed_wrap = allowed_same_page_column_wrap(number.box, implementation.box, models)
                if number_col != implementation_col and not allowed_wrap:
                    issues.append(
                        make_issue(
                            "number_implementation_different_columns",
                            "error",
                            entry,
                            "Certificate number and implementation are in different inferred columns outside the allowed bottom-left to top-right wrap pattern.",
                            models,
                            {
                                "number_column": number_col,
                                "implementation_column": implementation_col,
                                "page_split_x": round(model.split_x, 6),
                            },
                        )
                    )
                if implementation.box.y0 + args.vertical_tolerance < number.box.y0 and not allowed_wrap:
                    issues.append(
                        make_issue(
                            "implementation_above_number",
                            "error",
                            entry,
                            "Implementation appears above its certificate number outside the allowed column-wrap exception.",
                            models,
                            {
                                "number_y0": round(number.box.y0, 6),
                                "implementation_y0": round(implementation.box.y0, 6),
                                "vertical_tolerance": args.vertical_tolerance,
                            },
                        )
                    )
                vertical_gap = implementation.box.y0 - number.box.y1
                if (
                    number_col == implementation_col
                    and vertical_gap > args.max_number_to_implementation_gap
                    and not allowed_wrap
                ):
                    issues.append(
                        make_issue(
                            "number_implementation_large_gap",
                            "warning",
                            entry,
                            "Large vertical gap between certificate number and implementation in the same column; inspect for a mismatched pair.",
                            models,
                            {
                                "vertical_gap": round(vertical_gap, 6),
                                "max_allowed_gap": args.max_number_to_implementation_gap,
                            },
                        )
                    )

    return issues


def process_json_file(
    path: Path,
    rel_path: str,
    args: argparse.Namespace,
    volume_reference_numbers: Dict[str, set],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    volume_key = volume_key_from_path(rel_path)
    pages = data.get("pages") or []
    raw_entities = data.get("entities") or []
    entries = [
        entry_from_entity(rel_path, index, entity, pages)
        for index, entity in enumerate(raw_entities)
        if entity.get("type") == "Entry"
    ]
    original_entry_count = len(entries)
    dedupe_stats = {
        "duplicates_removed": 0,
        "deduplicated_entry_count": len(entries),
        "duplicates_by_page": {},
        "duplicate_examples": [],
    }
    if not args.keep_duplicates:
        entries, dedupe_stats = remove_exact_duplicate_entries(entries)

    deduplicated_entry_count = len(entries)
    index_stats = {
        "standalone_index_numbers_ignored": 0,
        "ignored_index_numbers_by_page": {},
        "ignored_index_number_examples": [],
    }
    if not args.keep_standalone_index_numbers:
        entries, index_stats = remove_standalone_index_numbers(entries, volume_key, volume_reference_numbers)

    models = infer_page_models(entries, len(pages))
    paired = continuation_pairs(entries, models)

    issues = []
    for entry in entries:
        issues.extend(check_entry(entry, models, paired, args))
    issues.extend(implementation_to_next_entry_gap_issues(entries, models, args))
    issues.extend(implementation_value_gap_issues(entries, models, args))
    issues.extend(spacing_issues(entries, models, args))
    issues.extend(page_top_left_start_issues(entries, models, args))
    issues.extend(adjacent_missing_field_issues(entries, models, paired))

    if not args.include_info:
        issues = [issue for issue in issues if issue.get("severity") != "info"]

    prop_counter = Counter()
    for entry in entries:
        for field_type in FIELD_TYPES:
            if entry.field(field_type):
                prop_counter[field_type] += 1

    summary = {
        "json_path": rel_path,
        "volume_key": volume_key,
        "page_count": len(pages),
        "entry_count": len(entries),
        "original_entry_count": original_entry_count,
        "deduplicated_entry_count": deduplicated_entry_count,
        "duplicates_removed": dedupe_stats["duplicates_removed"],
        "duplicates_by_page": dedupe_stats["duplicates_by_page"],
        "duplicate_examples": dedupe_stats["duplicate_examples"],
        "standalone_index_numbers_ignored": index_stats["standalone_index_numbers_ignored"],
        "ignored_index_numbers_by_page": index_stats["ignored_index_numbers_by_page"],
        "ignored_index_number_examples": index_stats["ignored_index_number_examples"],
        "volume_reference_number_count": len(volume_reference_numbers.get(volume_key, set())),
        "field_counts": dict(prop_counter),
        "issue_count": len(issues),
        "issues_by_code": dict(Counter(issue["issue_code"] for issue in issues)),
        "issues_by_severity": dict(Counter(issue["severity"] for issue in issues)),
        "allowed_cross_page_continuation_pairs": len(paired) // 2,
    }
    return issues, summary


def iter_json_files(input_dir: Path, glob_pattern: str, max_files: Optional[int] = None) -> Iterable[Path]:
    count = 0
    for path in sorted(input_dir.glob(glob_pattern)):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        rel_parts = path.relative_to(input_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        yield path
        count += 1
        if max_files is not None and count >= max_files:
            return


def flatten_for_csv(issue: Dict[str, Any]) -> Dict[str, Any]:
    field_bboxes = issue.get("field_bboxes") or {}
    row = {
        "severity": issue.get("severity"),
        "issue_code": issue.get("issue_code"),
        "message": issue.get("message"),
        "json_path": issue.get("json_path"),
        "page_number": issue.get("page_number"),
        "column": issue.get("column"),
        "entry_index": issue.get("entry_index"),
        "entry_id": issue.get("entry_id"),
        "certificate_number": issue.get("certificate_number"),
        "certificate_implementation": issue.get("certificate_implementation"),
        "certificate_value": issue.get("certificate_value"),
        "entry_bbox": json.dumps(issue.get("entry_bbox"), ensure_ascii=False, sort_keys=True),
        "number_bbox": json.dumps(field_bboxes.get(FIELD_NUMBER), ensure_ascii=False, sort_keys=True),
        "implementation_bbox": json.dumps(field_bboxes.get(FIELD_IMPLEMENTATION), ensure_ascii=False, sort_keys=True),
        "value_bbox": json.dumps(field_bboxes.get(FIELD_VALUE), ensure_ascii=False, sort_keys=True),
        "evidence": json.dumps(issue.get("evidence") or {}, ensure_ascii=False, sort_keys=True),
    }
    return row


def write_reports(
    output_dir: Path,
    prefix: str,
    issues: Sequence[Dict[str, Any]],
    file_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}.issues.jsonl"
    csv_path = output_dir / f"{prefix}.issues.csv"
    summary_path = output_dir / f"{prefix}.summary.json"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for issue in issues:
            fh.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")

    fieldnames = [
        "severity",
        "issue_code",
        "message",
        "json_path",
        "page_number",
        "column",
        "entry_index",
        "entry_id",
        "certificate_number",
        "certificate_implementation",
        "certificate_value",
        "entry_bbox",
        "number_bbox",
        "implementation_bbox",
        "value_bbox",
        "evidence",
    ]
    # Use utf-8-sig so spreadsheet apps such as Excel/Numbers reliably detect
    # Cyrillic text as UTF-8 instead of opening it as mojibake.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(flatten_for_csv(issue))

    summary = {
        "total_issue_count": len(issues),
        "issues_by_code": dict(Counter(issue["issue_code"] for issue in issues)),
        "issues_by_severity": dict(Counter(issue["severity"] for issue in issues)),
        "total_original_entry_count": sum((item.get("original_entry_count") or 0) for item in file_summaries),
        "total_deduplicated_entry_count": sum((item.get("deduplicated_entry_count") or 0) for item in file_summaries),
        "total_analysis_entry_count": sum((item.get("entry_count") or 0) for item in file_summaries),
        "total_duplicates_removed": sum((item.get("duplicates_removed") or 0) for item in file_summaries),
        "total_standalone_index_numbers_ignored": sum(
            (item.get("standalone_index_numbers_ignored") or 0) for item in file_summaries
        ),
        "files": list(file_summaries),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)

    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect likely OCR and pairing errors in patent certificate JSON exports."
    )
    parser.add_argument("--input", default=".", help="Directory containing train/test/unassigned JSON exports.")
    parser.add_argument("--output", default="output/ocr_error_reports", help="Directory for non-mutating reports.")
    parser.add_argument("--glob", default="**/*.json", help="Glob under --input for source JSON files.")
    parser.add_argument("--prefix", default="ocr_error_report", help="Output file prefix.")
    parser.add_argument("--max-files", type=int, default=None, help="Debug option: process only the first N JSON files.")
    parser.add_argument("--include-info", action="store_true", help="Include info-level issues in reports.")
    parser.add_argument(
        "--emit-allowed-continuations",
        action="store_true",
        help="Emit info rows for allowed bottom-right to next-page top-left continuation pairs.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not remove exact same-page text duplicate entries before diagnostics.",
    )
    parser.add_argument(
        "--keep-standalone-index-numbers",
        action="store_true",
        help="Do not ignore standalone number-only entries that also appear with text elsewhere in the same volume.",
    )

    parser.add_argument("--min-certificate-number-length", type=int, default=5)
    parser.add_argument("--max-certificate-number-length", type=int, default=7)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--implementation-digit-ratio", type=float, default=0.22)
    parser.add_argument("--min-implementation-chars", type=int, default=48)

    parser.add_argument("--vertical-tolerance", type=float, default=0.01)
    parser.add_argument("--max-number-to-implementation-gap", type=float, default=0.22)
    parser.add_argument("--wide-field-width-ratio", type=float, default=1.8)

    parser.add_argument("--mad-multiplier", type=float, default=4.0)
    parser.add_argument("--start-delta-ratio", type=float, default=1.85)
    parser.add_argument("--whitespace-gap-ratio", type=float, default=2.4)
    parser.add_argument("--min-large-start-delta", type=float, default=0.075)
    parser.add_argument("--min-large-whitespace-gap", type=float, default=0.045)
    parser.add_argument("--implementation-to-next-gap-ratio", type=float, default=1.5)
    parser.add_argument("--min-implementation-to-next-gap", type=float, default=0.045)
    parser.add_argument("--implementation-gap-min-baseline-pairs", type=int, default=3)
    parser.add_argument("--implementation-value-gap-ratio", type=float, default=2.4)
    parser.add_argument("--min-implementation-value-gap", type=float, default=0.03)
    parser.add_argument("--implementation-value-gap-min-baseline-pairs", type=int, default=3)
    parser.add_argument("--first-entry-top-absolute-threshold", type=float, default=0.18)
    parser.add_argument("--first-entry-top-relative-delta", type=float, default=0.08)
    parser.add_argument("--first-entry-min-page-rows", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    issues: List[Dict[str, Any]] = []
    file_summaries: List[Dict[str, Any]] = []

    paths = list(iter_json_files(input_dir, args.glob, args.max_files))
    if not paths:
        raise SystemExit(f"No JSON files found under {input_dir} with glob {args.glob!r}.")

    reference_paths = list(iter_json_files(input_dir, "**/*.json"))
    volume_reference_numbers, reference_stats = build_volume_reference_numbers(input_dir, reference_paths)
    print(
        "Built same-volume reference numbers from "
        f"{reference_stats['reference_files_scanned']} JSON files "
        f"across {reference_stats['volume_count']} volumes."
    )

    for index, path in enumerate(paths, start=1):
        rel_path = str(path.relative_to(input_dir))
        try:
            file_issues, file_summary = process_json_file(path, rel_path, args, volume_reference_numbers)
        except Exception as exc:  # Keep one bad file from stopping a corpus scan.
            file_issues = [
                {
                    "severity": "error",
                    "issue_code": "json_processing_failed",
                    "message": f"Could not process JSON file: {exc}",
                    "json_path": rel_path,
                    "entry_index": None,
                    "entry_id": None,
                    "page_number": None,
                    "column": None,
                    "entry_bbox": None,
                    "field_bboxes": {},
                    "field_columns": {},
                    "field_confidence": {},
                    "certificate_number": "",
                    "certificate_implementation": "",
                    "certificate_value": "",
                    "evidence": {"exception_type": type(exc).__name__},
                }
            ]
            file_summary = {
                "json_path": rel_path,
                "volume_key": volume_key_from_path(rel_path),
                "page_count": None,
                "entry_count": None,
                "deduplicated_entry_count": None,
                "original_entry_count": None,
                "duplicates_removed": 0,
                "duplicates_by_page": {},
                "duplicate_examples": [],
                "standalone_index_numbers_ignored": 0,
                "ignored_index_numbers_by_page": {},
                "ignored_index_number_examples": [],
                "volume_reference_number_count": 0,
                "field_counts": {},
                "issue_count": 1,
                "issues_by_code": {"json_processing_failed": 1},
                "issues_by_severity": {"error": 1},
                "allowed_cross_page_continuation_pairs": 0,
            }
        issues.extend(file_issues)
        file_summaries.append(file_summary)
        if index % 10 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)} files; issues so far: {len(issues)}")

    issues.sort(
        key=lambda issue: (
            issue.get("json_path") or "",
            issue.get("page_number") or 0,
            issue.get("column") if issue.get("column") is not None else 9,
            issue.get("entry_index") if issue.get("entry_index") is not None else 10**12,
            -SEVERITY_ORDER.get(issue.get("severity"), 0),
            issue.get("issue_code") or "",
        )
    )
    paths_written = write_reports(output_dir, args.prefix, issues, file_summaries)
    summary_path = paths_written["summary"]
    with summary_path.open("r", encoding="utf-8") as fh:
        summary_data = json.load(fh)
    summary_data["volume_reference_stats"] = reference_stats
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print("Done.")
    for label, path in paths_written.items():
        print(f"{label}: {path}")
    print(f"total issues: {len(issues)}")
    for severity, count in Counter(issue["severity"] for issue in issues).most_common():
        print(f"{severity}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
