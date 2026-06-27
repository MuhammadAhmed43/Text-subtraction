"""Layout Graph Builder - merges OCR blocks, clusters duplicates, links labels to values."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def bbox_overlap(a: Dict, b: Dict) -> float:
    """Compute IoU (intersection-over-union) between two bboxes."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def text_similarity(a: str, b: str) -> float:
    """Simple normalized text similarity (character-level Jaccard)."""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if not a_lower or not b_lower:
        return 0.0
    if a_lower == b_lower:
        return 1.0
    set_a = set(a_lower)
    set_b = set(b_lower)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def classify_block_type(text: str) -> str:
    """Heuristic classification of a text block as label, value, or other."""
    text_stripped = text.strip()
    if not text_stripped:
        return "noise"

    # Short text ending with colon -> label
    if text_stripped.endswith(":") and len(text_stripped) < 60:
        return "label"

    # Detect form headers / document titles (all-caps, long, or known keywords)
    header_patterns = [
        r"(?i)^(account\s*opening|opening\s*form|application\s*form)",
        r"(?i)^(for\s+bank\s+use|office\s+use\s+only|branch\s+use)",
    ]
    for hp in header_patterns:
        if re.search(hp, text_stripped):
            return "header"  # Treat as noise
    # All-uppercase word-clusters that are clearly headings (e.g. ACCOUNTOPENINGAPPLICATION)
    if re.match(r'^[A-Z][A-Z/\s]{8,}$', text_stripped) and not any(c.islower() for c in text_stripped):
        return "header"

    # Common form label patterns
    label_patterns = [
        r"(?i)^(name|date|address|phone|email|cnic|account|branch|city|"
        r"father|mother|occupation|nationality|religion|gender|dob|"
        r"title|mr|mrs|ms|sr|jr|signature|witness|nominee|"
        r"applicant|customer|deposit|amount|balance|type|status|"
        r"mobile|cell|fax|zip|postal|code|no\.?|number|#)",
    ]
    for pattern in label_patterns:
        if re.search(pattern, text_stripped):
            if len(text_stripped) < 50:
                return "label"

    # Checkbox indicators
    if text_stripped in ("X", "x", "yes", "no", "true", "false"):
        return "checkbox"

    # Numbers that look like values
    if re.match(r"^[\d\s\-\./,]+$", text_stripped) and len(text_stripped) > 2:
        return "value"

    # Longer text is more likely a value
    if len(text_stripped) > 40:
        return "value"

    return "text"


def merge_ocr_blocks(
    blocks_by_model: Dict[str, List[Dict]],
) -> List[Dict[str, Any]]:
    """Merge OCR blocks from multiple models into unified clusters.
    
    Args:
        blocks_by_model: {model_name: [blocks]} where each block has
            text, bbox, confidence, model
            
    Returns:
        List of merged nodes with source_blocks references
    """
    all_blocks: List[Dict] = []
    for model_name, blocks in blocks_by_model.items():
        for i, block in enumerate(blocks):
            block["block_id"] = f"blk_{model_name}_{i:04d}"
            all_blocks.append(block)

    if not all_blocks:
        return []

    # Sort by y1 then x1 (reading order)
    all_blocks.sort(key=lambda b: (b["bbox"]["y1"], b["bbox"]["x1"]))

    # Cluster overlapping blocks
    clusters: List[List[Dict]] = []
    used = set()

    for i, block_a in enumerate(all_blocks):
        if i in used:
            continue
        cluster = [block_a]
        used.add(i)

        for j, block_b in enumerate(all_blocks):
            if j in used:
                continue
            # Check spatial overlap + text similarity
            iou = bbox_overlap(block_a["bbox"], block_b["bbox"])
            tsim = text_similarity(block_a["text"], block_b["text"])

            if iou > 0.3 or (iou > 0.1 and tsim > 0.5):
                cluster.append(block_b)
                used.add(j)

        clusters.append(cluster)

    # Build merged nodes
    nodes: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        # Pick best text (highest confidence)
        best = max(cluster, key=lambda b: b.get("confidence", 0))
        # Merge bbox (union)
        x1 = min(b["bbox"]["x1"] for b in cluster)
        y1 = min(b["bbox"]["y1"] for b in cluster)
        x2 = max(b["bbox"]["x2"] for b in cluster)
        y2 = max(b["bbox"]["y2"] for b in cluster)

        avg_conf = sum(b.get("confidence", 0) for b in cluster) / len(cluster)
        block_type = classify_block_type(best["text"])

        node = {
            "node_id": f"node_{idx:04d}",
            "node_type": block_type,
            "text": best["text"],
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "confidence": avg_conf,
            "source_blocks": [b["block_id"] for b in cluster],
            "source_models": list(set(b["model"] for b in cluster)),
            "reading_order": idx,
        }
        nodes.append(node)

    return nodes


def link_labels_to_values(nodes: List[Dict]) -> List[Dict[str, Any]]:
    """Link label nodes to nearby value nodes using geometry.
    
    Returns list of edges with from, to, edge_type, score.
    """
    labels = [n for n in nodes if n["node_type"] == "label"]
    values = [n for n in nodes if n["node_type"] in ("value", "text")]
    edges: List[Dict[str, Any]] = []

    for label in labels:
        lb = label["bbox"]
        label_cx = (lb["x1"] + lb["x2"]) / 2
        label_cy = (lb["y1"] + lb["y2"]) / 2
        label_h = lb["y2"] - lb["y1"]
        label_w = lb["x2"] - lb["x1"]

        best_score = 0.0
        best_value = None
        best_edge_type = "near"

        for value in values:
            # CRITICAL: never link a label to another label node
            if value["node_type"] == "label":
                continue

            vb = value["bbox"]
            value_cx = (vb["x1"] + vb["x2"]) / 2
            value_cy = (vb["y1"] + vb["y2"]) / 2

            # Right-of: value starts near label's right edge, similar y
            if vb["x1"] >= lb["x2"] - 20:
                dy = abs(label_cy - value_cy)
                dx = vb["x1"] - lb["x2"]
                if dy < label_h * 1.5 and dx < label_w * 4:
                    score = max(0, 1.0 - (dx / 500) - (dy / 200))
                    if score > best_score:
                        best_score = score
                        best_value = value
                        best_edge_type = "right_of"

            # Below: value is directly below label
            if vb["y1"] >= lb["y2"] - 10:
                dy = vb["y1"] - lb["y2"]
                dx = abs(label_cx - value_cx)
                if dy < label_h * 3 and dx < label_w * 1.5:
                    score = max(0, 0.9 - (dy / 300) - (dx / 400))
                    if score > best_score:
                        best_score = score
                        best_value = value
                        best_edge_type = "below"

        if best_value and best_score > 0.2:
            edges.append({
                "from": label["node_id"],
                "to": best_value["node_id"],
                "edge_type": best_edge_type,
                "score": round(best_score, 3),
            })

    return edges


def build_layout_graph(
    blocks_by_model: Dict[str, List[Dict]],
    page_id: str,
    document_id: str,
) -> Dict[str, Any]:
    """Build the full layout graph for a page.
    
    Returns the layout graph dict with nodes and edges.
    """
    nodes = merge_ocr_blocks(blocks_by_model)
    edges = link_labels_to_values(nodes)

    return {
        "document_id": document_id,
        "page_id": page_id,
        "nodes": nodes,
        "edges": edges,
    }
