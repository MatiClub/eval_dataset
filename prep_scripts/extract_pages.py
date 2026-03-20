#!/usr/bin/env python3
import sys
from pypdf import PdfReader, PdfWriter

def parse_range(r):
    if "-" in r:
        a, b = r.split("-")
        return int(a), int(b)
    p = int(r)
    return p, p

if len(sys.argv) < 4:
    print("usage: pdf_split_ranges.py input.pdf output_prefix 5-8 11-12 13")
    sys.exit(1)

inp = sys.argv[1]
prefix = sys.argv[2]
ranges = sys.argv[3:]

reader = PdfReader(inp)

for r in ranges:
    start, end = parse_range(r)
    writer = PdfWriter()

    for p in range(start-1, end):
        writer.add_page(reader.pages[p])

    out = f"{prefix}_{start}-{end}.pdf"
    with open(out, "wb") as f:
        writer.write(f)

    print(f"written {out}")