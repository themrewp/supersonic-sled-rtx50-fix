"""Build the 1300x372 Nexus Mods header from in-game captures (2304x1296).

usage: python make_header.py <ignition.png> <smoketrail.png> <out_dir>
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1300, 372
FONTS = "C:/Windows/Fonts/"
ORANGE = (255, 150, 40)
CREAM = (255, 240, 215)


def font(name, size, variation=None):
    f = ImageFont.truetype(FONTS + name, size)
    if variation:
        f.set_variation_by_name(variation)
    return f


def shadow_text(img, xy, text, f, fill, blur=6, anchor="la"):
    """Text with a soft dark glow so it reads on busy backgrounds."""
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).text(xy, text, font=f, fill=(0, 0, 0, 230), anchor=anchor,
                              stroke_width=4, stroke_fill=(0, 0, 0, 230))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).text(xy, text, font=f, fill=fill, anchor=anchor)


def text_block(img, x, y, anchor_right=False):
    a = "ra" if anchor_right else "la"
    shadow_text(img, (x, y), "SUPERSONIC SLED", font("impact.ttf", 92), CREAM, anchor=a)
    shadow_text(img, (x, y + 104), "RTX 50-SERIES FIX", font("bahnschrift.ttf", 46, "Bold"), ORANGE, anchor=a)
    shadow_text(img, (x, y + 162), "Runs again on RTX 50 cards. No second GPU needed.",
                font("bahnschrift.ttf", 24, "SemiBold"), CREAM, blur=4, anchor=a)


def ignition(src, out):
    # crop clear of the game's HUD (thrust bar top, mach dial top-right, buttons bottom)
    shot = Image.open(src).convert("RGBA").crop((0, 110, 1900, 1015))
    shot = shot.resize((round(shot.width * H / shot.height), H), Image.LANCZOS)
    img = Image.new("RGBA", (W, H), (22, 14, 10, 255))
    x0 = W - shot.width
    img.alpha_composite(shot, (x0, 0))
    # feather the left edge of the shot into the dark panel
    fade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(fade)
    feather = 150
    for i in range(feather):
        alpha = int(255 * (1 - i / feather) ** 1.6)
        d.line([(x0 + i, 0), (x0 + i, H)], fill=(22, 14, 10, alpha))
    img.alpha_composite(fade)
    text_block(img, 40, 72)
    img.convert("RGB").save(out, quality=95)


def smoke(src, out):
    shot = Image.open(src).convert("RGBA").crop((0, 390, 2220, 1025))
    img = shot.resize((W, H), Image.LANCZOS)
    # darken the lower right where the text sits
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(shade)
    for x in range(W):
        t = max(0.0, (x - 560) / (W - 560))
        d.line([(x, 0), (x, H)], fill=(20, 8, 4, int(170 * t ** 1.3)))
    img.alpha_composite(shade)
    text_block(img, W - 40, 82, anchor_right=True)
    img.convert("RGB").save(out, quality=95)


if __name__ == "__main__":
    ign, smk, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    ignition(ign, os.path.join(out_dir, "header_ignition.png"))
    smoke(smk, os.path.join(out_dir, "header_smoketrail.png"))
    print("ok")
