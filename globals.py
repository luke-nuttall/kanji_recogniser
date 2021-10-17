from pathlib import Path

with open("kanji.txt", "r") as fp:
    ALL_KANJI = list(fp.read().strip())

# a dictionary for fast lookups of the index given the kanji
# this is necessary to ensure consistent labelling when shuffling the kanji list
KANJI_INDICES = {kanji: ii for ii, kanji in enumerate(ALL_KANJI)}

ALL_FONTS = list(Path("fonts").glob("*.otf"))

IMG_SIZE = 32
CATEGORIES_ANGLE = 36
CATEGORIES_KANJI = len(ALL_KANJI)
