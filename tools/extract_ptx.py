"""Pull the PTX (GPU code) embedded in SupersonicSled.exe into ptx/*.ptx.

usage: python extract_ptx.py "D:\\SuperSonicSled\\bin\\SupersonicSled.exe" ptx
"""
import os
import re
import sys

exe, out_dir = sys.argv[1], sys.argv[2]
data = open(exe, "rb").read()
os.makedirs(out_dir, exist_ok=True)
n = 0
for m in re.finditer(rb"\t\.version ", data):
    start = m.start()
    while data[start - 1] != 0:
        start -= 1
    text = data[start:data.index(b"\0", start)].decode("latin1")
    src = re.search(r'"(\w+\.cu)"', text)
    tgt = re.search(r"\.target (\w+)", text)
    name = f"{n:02d}_{src.group(1) if src else 'x'}_{tgt.group(1) if tgt else ''}.ptx"
    with open(os.path.join(out_dir, name), "w", encoding="latin1") as f:
        f.write(text)
    print(name)
    n += 1
