#!/bin/sh
# Builds the three 32-bit replacement DLLs into ./out using zig's bundled clang.
# usage: ZIG=path/to/zig.exe ./build.sh "D:/SuperSonicSled/bin/SupersonicSled.exe"
set -e
ZIG="${ZIG:-zig}"
EXE="${1:-D:/SuperSonicSled/bin/SupersonicSled.exe}"
cd "$(dirname "$0")"
mkdir -p out
python tools/extract_ptx.py "$EXE" ptx
python tools/ptx2c.py src/kernels_gen.c ptx/*.ptx
CF="-target x86-windows-gnu -O2 -fwrapv -fno-strict-aliasing -ffp-contract=off -Wall -Wno-unused-label -Wno-unused-variable -Wno-unused-but-set-variable"
$ZIG cc $CF -shared -o out/cudart32_30_11.dll src/cudart.c src/kernels_gen.c src/cudart.def -ld3d11
$ZIG cc $CF -shared -o out/cufft32_30_11.dll src/cufft.c src/cufft.def
$ZIG cc $CF -shared -o out/nvcuda.dll src/nvcuda.c src/nvcuda.def
ls -la out
