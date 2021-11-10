import datagen
from matplotlib import pyplot as plt, font_manager
from rendering import gen_raw_sample
from globals import IMG_SIZE, ALL_FONTS, ALL_KANJI, CATEGORIES_ANGLE
import time
from contextlib import contextmanager
from random import choice, randint, random
from math import pi

font_manager.fontManager.addfont("fonts/NotoSansJP-Regular.otf")
plt.rc('font', family='Noto Sans JP')

@contextmanager
def timer(label):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(label, elapsed)


def test_many():
    renderer = datagen.RustRenderer(IMG_SIZE, [str(x) for x in ALL_FONTS], ALL_KANJI, CATEGORIES_ANGLE)

    iterations = 10_000

    with timer(f"rust {iterations} iterations:"):
        for i in range(iterations):
            renderer.render()

    with timer(f"pillow {iterations} iterations:"):
        for i in range(iterations):
            gen_raw_sample(ALL_KANJI, ALL_FONTS)


def test_one():
    renderer = datagen.RustRenderer(IMG_SIZE, [str(x) for x in ALL_FONTS], ALL_KANJI, CATEGORIES_ANGLE)
    renderer.kanji = ALL_KANJI[:20]
    renderer.kanji = ALL_KANJI[250:260]
    img, kanji, font_size, angle = renderer.render()
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    print(kanji, ALL_KANJI[kanji], font_size, angle)
    plt.show()


def test_batch():
    renderer = datagen.RustRenderer(IMG_SIZE, [str(x) for x in ALL_FONTS], ALL_KANJI, CATEGORIES_ANGLE)
    renderer.kanji = ALL_KANJI[:20]
    renderer.kanji = ALL_KANJI[250:260]
    n_rows = 4
    n_cols = 8
    scale = 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale), constrained_layout=True)
    samples = renderer.render_batch(n_rows*n_cols)
    for ax, (img, kanji, font_size, angle) in zip(axes.flatten(), samples):
        ax: plt.Axes
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{ALL_KANJI[kanji]} ({kanji})")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.show()


def main():
    test_batch()


if __name__ == "__main__":
    main()
