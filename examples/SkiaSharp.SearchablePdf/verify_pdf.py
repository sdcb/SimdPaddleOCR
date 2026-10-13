"""Independent PDF text, geometry, image and rendering verification.

Requires: python -m pip install pymupdf pillow
Usage: python verify_pdf.py output.pdf source-folder --preview-dir tmp/pdfs
"""
import argparse
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def expected_characters(page_info):
    for line in page_info["Lines"]:
        for char in line["Characters"]:
            count = len(char["Text"])
            tl = (char["X1"], char["Y1"])
            tr = (char["X2"], char["Y2"])
            br = (char["X3"], char["Y3"])
            bl = (char["X4"], char["Y4"])
            if line["AppliedRotationDegrees"] == 180:
                tl, tr, br, bl = br, bl, tl, tr
            for i, text in enumerate(char["Text"]):
                def lerp(a, b, t):
                    return tuple(x + (y - x) * t for x, y in zip(a, b))
                points = [lerp(tl, tr, i / count), lerp(tl, tr, (i + 1) / count),
                          lerp(bl, br, (i + 1) / count), lerp(bl, br, i / count)]
                yield text, points


def verify(pdf_path, source_dir, preview_dir):
    report = json.loads(pdf_path.with_suffix(".ocr.json").read_text(encoding="utf-8"))
    scale = 72 / report["Dpi"]
    document = fitz.open(pdf_path)
    assert len(document) == len(report["Pages"]), "Page count mismatch"
    summary = {"Pages": len(document), "Characters": 0, "MaxBoxErrorPixels": 0,
               "OriginalJpegsPreserved": 0, "InvisibleText": True, "PreviewPages": []}
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    for page, info in zip(document, report["Pages"]):
        expected = list(expected_characters(info))
        expected_text = "".join(c for c, _ in expected)
        trace = page.get_texttrace()
        recovered = "".join(chr(c[0]) for span in trace for c in span["chars"])
        assert recovered == expected_text, f"Page {page.number + 1}: Unicode or reading order differs"
        assert all(span["type"] == 3 for span in trace), "Visible text was written"
        # A normal reader adds layout newlines / spaces; non-whitespace content
        # and order must also survive its default extraction, not just the stream.
        assert re.sub(r"\s", "", page.get_text()) == re.sub(r"\s", "", expected_text), \
            f"Page {page.number + 1}: default copy order differs"
        raw = page.get_text("rawdict", flags=fitz.TEXTFLAGS_RAWDICT | fitz.TEXT_INHIBIT_SPACES)
        actual = [c for b in raw["blocks"] if b["type"] == 0
                  for line in b["lines"] for span in line["spans"] for c in span["chars"]]
        assert "".join(c["c"] for c in actual) == expected_text, "Selection characters differ"
        for (text, points), char in zip(expected, actual):
            target = (min(p[0] for p in points), min(p[1] for p in points),
                      max(p[0] for p in points), max(p[1] for p in points))
            measured = tuple(v / scale for v in char["bbox"])
            error = max(abs(a - b) for a, b in zip(target, measured))
            summary["MaxBoxErrorPixels"] = max(summary["MaxBoxErrorPixels"], error)
            assert error <= 3, f"Page {page.number + 1}: character box differs by {error:.2f}px"
        summary["Characters"] += len(expected)
        images = page.get_images(full=True)
        assert len(images) == 1, "Expected exactly one scan image per page"
        image_xref = images[0][0]
        original = (source_dir / info["File"]).read_bytes()
        if original.startswith(b"\xff\xd8") and document.xref_stream_raw(image_xref) == original:
            summary["OriginalJpegsPreserved"] += 1
        assert abs(page.rect.width / scale - info["Width"]) < .01
        assert abs(page.rect.height / scale - info["Height"]) < .01
        if preview_dir and page.number in {0, 1, 2, len(document) // 2, len(document) - 1}:
            # Render both the real page and the same image on an otherwise empty
            # page. Exact pixel equality proves the text layer changes no pixels.
            pix = page.get_pixmap(dpi=110, alpha=False)
            reference = fitz.open()
            image_only = reference.new_page(width=page.rect.width, height=page.rect.height)
            image_only.insert_image(image_only.rect, stream=document.extract_image(image_xref)["image"])
            ref_pix = image_only.get_pixmap(dpi=110, alpha=False)
            assert pix.samples == ref_pix.samples, f"Page {page.number + 1}: text changes rendering"
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            image.save(preview_dir / f"page-{page.number + 1}.png")
            draw = ImageDraw.Draw(image)
            zoom = 110 / report["Dpi"]
            for _, points in expected:
                draw.polygon([(x * zoom, y * zoom) for x, y in points], outline=(220, 30, 40))
            image.save(preview_dir / f"alignment-{page.number + 1}.png")
            summary["PreviewPages"].append(page.number + 1)
            reference.close()
    summary["MaxBoxErrorPixels"] = round(summary["MaxBoxErrorPixels"], 4)
    document.close()
    print(json.dumps(summary, indent=2))
    pdf_path.with_suffix(".verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    verify(args.pdf, args.source_dir, args.preview_dir)
