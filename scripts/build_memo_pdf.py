#!/usr/bin/env python3
"""Render docs/technical-memo.md to a polished PDF for email.

The memo's mermaid diagram does not survive markdown->HTML conversion, so it
is swapped for a self-contained HTML/CSS pipeline diagram before rendering.
Chrome (headless) does the final HTML->PDF print.

Usage: .venv/bin/python scripts/build_memo_pdf.py
Output: dist/AutoAce-Technical-Memo.pdf
"""

import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "technical-memo.md"
OUT_DIR = ROOT / "dist"
HTML = OUT_DIR / "technical-memo.html"
PDF = OUT_DIR / "AutoAce-Technical-Memo.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Self-contained pipeline diagram (replaces the mermaid block).
DIAGRAM = """
<div class="pipeline">
  <div class="stage"><b>Upload</b><span>ZIP / folder</span></div>
  <div class="arrow">&rarr;</div>
  <div class="stage"><b>Validate</b><span>report before spend</span></div>
  <div class="arrow">&rarr;</div>
  <div class="stage local"><b>Local layer</b><span>$0/call &middot; VAD &middot; PANNs &middot; SQUIM</span></div>
  <div class="arrow">&rarr;</div>
  <div class="stage llm"><b>Gemini call</b><span>tone + intensity</span>
    <span class="fallback">API fail &rarr; local fallback</span></div>
  <div class="arrow">&rarr;</div>
  <div class="stage"><b>Fusion</b><span>9-field JSON</span></div>
  <div class="arrow">&rarr;</div>
  <div class="stage out"><b>Review</b><span>CSV / JSON download</span></div>
</div>
"""

CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  color: #1e293b; font-size: 10.5px; line-height: 1.5; max-width: 100%;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 21px; color: #0f172a; margin: 0 0 2px; letter-spacing: -.02em; }
h1 + p { color: #64748b; margin: 0 0 14px; font-size: 10px; }
h2 {
  font-size: 14px; color: #0f172a; margin: 22px 0 8px; padding-bottom: 4px;
  border-bottom: 2px solid #2563EB; letter-spacing: -.01em;
}
h3 { font-size: 11.5px; color: #334155; margin: 14px 0 6px; }
hr { border: 0; border-top: 1px solid #e2e8f0; margin: 14px 0; }
p { margin: 6px 0; }
ul { margin: 6px 0; padding-left: 18px; }
li { margin: 3px 0; }
strong { color: #0f172a; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9.5px;
  background: #f1f5f9; padding: 1px 4px; border-radius: 3px; color: #0f172a;
}
a { color: #2563EB; text-decoration: none; }

/* callout blockquote (the n=3 caveat) */
blockquote {
  margin: 10px 0; padding: 8px 12px; background: #eff6ff;
  border-left: 3px solid #2563EB; border-radius: 4px; color: #1e3a8a;
}
blockquote p { margin: 3px 0; }

table {
  border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5px;
  page-break-inside: avoid;
}
th {
  background: #0f172a; color: #fff; text-align: left; padding: 5px 8px;
  font-weight: 600; font-size: 9px; text-transform: uppercase;
  letter-spacing: .03em;
}
td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }

/* exec-summary checkmarks read green; ✗ / weakness dashes read muted-red */
.lead li { list-style: none; }
.lead ul { padding-left: 4px; }

h2 { page-break-after: avoid; }
h3 { page-break-after: avoid; }

/* pipeline diagram */
.pipeline {
  display: flex; align-items: stretch; gap: 4px; margin: 12px 0 16px;
  flex-wrap: nowrap;
}
.pipeline .stage {
  flex: 1; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 6px;
  text-align: center; background: #f8fafc; display: flex; flex-direction: column;
  justify-content: center; min-height: 54px;
}
.pipeline .stage b { font-size: 9.5px; color: #0f172a; display: block; }
.pipeline .stage span { font-size: 8px; color: #64748b; display: block; margin-top: 2px; }
.pipeline .stage.local { background: #ecfdf5; border-color: #6ee7b7; }
.pipeline .stage.llm { background: #eff6ff; border-color: #93c5fd; }
.pipeline .stage.out { background: #f5f3ff; border-color: #c4b5fd; }
.pipeline .stage .fallback { color: #b45309; font-size: 7.5px; margin-top: 3px; }
.pipeline .arrow { align-self: center; color: #94a3b8; font-size: 14px; }
"""


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    text = SRC.read_text(encoding="utf-8")

    # Drop the fenced mermaid block; we inject a styled HTML diagram instead.
    text = re.sub(r"```mermaid.*?```", "@@DIAGRAM@@", text, flags=re.DOTALL)

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<p>@@DIAGRAM@@</p>", DIAGRAM).replace("@@DIAGRAM@@", DIAGRAM)

    # Tag the executive-summary bullet lists so the checkmarks sit flush.
    body = body.replace("<h2>Executive summary</h2>", '<h2>Executive summary</h2><div class="lead">')
    body = body.replace("<h2>1. Architecture</h2>", "</div><h2>1. Architecture</h2>")

    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML.write_text(html, encoding="utf-8")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True,
    )
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
