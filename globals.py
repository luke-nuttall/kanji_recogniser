from pathlib import Path

with open("kanji.txt", "r") as fp:
    ALL_KANJI = list(fp.read().strip())

ALL_FONTS = list(Path("fonts").glob("*.otf"))

IMG_SIZE = 32
CATEGORIES_ANGLE = 36
CATEGORIES_KANJI = len(ALL_KANJI)
