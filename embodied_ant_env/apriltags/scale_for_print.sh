#!/bin/bash

# Note: The width is for tag physical size. The detected size is the interior of the rectangle.
# for tagCircle21h7, the inside is 5 out of the 9 pixels so 5/9 of the physical size.

for f in "$@"; do
    outname="$(dirname "$f")/$(basename "$f" | sed 's/^tag/marker/')"
    printf "Processing %s\n" "$f"

    # # 590 is pixel size for 50mm at 300 DPI: 50mm / 25.4 * 300 ≈ 590
    # magick "$f" -units PixelsPerInch -density 300 -filter point -scale 590x590\! "${outname%.png}_50mm.png"

    # 957 is pixel size for 81mm at 300 DPI: 81mm / 25.4 * 300 ≈ 957
    magick "$f" -units PixelsPerInch -density 300 -filter point -scale 957x957\! "${outname%.png}_81mm.png"

    # 2126 is pixel size for 180mm at 300 DPI: 180mm / 25.4 * 300 ≈ 2126
    magick "$f" -units PixelsPerInch -density 300 -filter point -scale 2126x2126\! "${outname%.png}_180mm.png"

done
