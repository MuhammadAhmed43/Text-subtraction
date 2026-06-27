"""Quick test: render one PDF and check output."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
from ocr_extract.config import load_config
from ocr_extract.input_manager import discover_inputs, build_manifest
import json

cfg = load_config("configs/default.yaml")
cfg.work_path = Path("work")

# Test with just one file
files = discover_inputs(Path("AOF_01.pdf"))
print(f"Found {len(files)} files")

manifest = build_manifest(files, cfg)
docs = manifest["documents"]
print(f"Documents: {len(docs)}")
for doc in docs:
    print(f"  {doc['source_file']}: {doc['page_count']} pages, status={doc['status']}")
    for p in doc["pages"]:
        print(f"    Page {p['page_number']}: {p['width']}x{p['height']} @ {p['dpi']}dpi")
        print(f"    Image: {p['image_path']}")

print("\nManifest saved to work/manifests/document_manifest.json")
