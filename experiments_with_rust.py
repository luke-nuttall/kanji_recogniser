import datagen
from matplotlib import pyplot as plt
from rendering import gen_raw_sample
from globals import IMG_SIZE, ALL_FONTS, ALL_KANJI
import time
from contextlib import contextmanager
from random import choice, randint, random
from math import pi


@contextmanager
def timer(label):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(label, elapsed)


def main():
    renderer = datagen.RustRenderer(IMG_SIZE, [str(x) for x in ALL_FONTS])
    renderer.populate_cache(ALL_KANJI)

    iterations = 10_000

    with timer(f"rust {iterations} iterations:"):
        for i in range(iterations):
            kanji = choice(ALL_KANJI)
            font_size = randint(16, 24)
            angle = random()*10 - 5
            arr = renderer.render(kanji, font_size, angle*pi/180)

    with timer(f"pillow {iterations} iterations:"):
        for i in range(iterations):
            gen_raw_sample(ALL_KANJI, ALL_FONTS)

    plt.imshow(arr, cmap="gray", vmin=0, vmax=1)
    print(kanji, font_size, angle)
    plt.show()


if __name__ == "__main__":
    main()
