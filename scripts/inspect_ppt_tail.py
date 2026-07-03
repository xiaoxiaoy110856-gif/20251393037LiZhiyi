from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def main() -> None:
    ppt = Path(sys.argv[1])
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    with zipfile.ZipFile(ppt) as zf:
        slide_names = sorted(
            [
                name
                for name in zf.namelist()
                if re.match(r"ppt/slides/slide\d+\.xml$", name)
            ],
            key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
        )
        print(f"slides={len(slide_names)}")
        for name in slide_names[-tail:]:
            root = ET.fromstring(zf.read(name))
            texts = [
                (node.text or "").strip()
                for node in root.findall(".//a:t", NS)
                if (node.text or "").strip()
            ]
            joined = " | ".join(texts)
            print(f"{name}: {joined[:900]}")


if __name__ == "__main__":
    main()
