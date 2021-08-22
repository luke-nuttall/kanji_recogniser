from PIL import Image, ImageDraw, ImageFont
from random import choice, randint, random
import numpy as np
from collections import namedtuple

from globals import IMG_SIZE


RawSample = namedtuple("RawSample", ["image", "kanji_index", "font_size", "angle"])


WHITE = (255, 255, 255)


def render_kanji(kanji, fontpath, fsize, angle):
    color_text = (0, 0, 0, randint(128, 255))
    color_bg = (255, 255, 255, randint(128, 255))
    bgsize = randint(fsize, IMG_SIZE)
    off_y = fsize * 0.3
    circle_min = IMG_SIZE//2 - bgsize//2
    circle_max = IMG_SIZE//2 + bgsize//2
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE))
    font = ImageFont.truetype(str(fontpath), fsize)
    draw = ImageDraw.Draw(img)
    draw.ellipse((circle_min, circle_min, circle_max, circle_max), color_bg)
    text_pos = IMG_SIZE//2 - fsize//2
    draw.text((text_pos, text_pos-off_y), kanji, color_text, font=font)
    return img.rotate(angle, resample=Image.BILINEAR)


def render_clutter():
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)
    for _ in range(randint(0, 100)):
        x, y = randint(-IMG_SIZE, IMG_SIZE*2), randint(-IMG_SIZE, IMG_SIZE*2)
        sx, sy = randint(-IMG_SIZE, IMG_SIZE*2), randint(-IMG_SIZE, IMG_SIZE*2)
        width = randint(1, 20)
        color = (0, 0, 0, randint(1, 255))
        draw_type = randint(0, 4)
        if draw_type == 0:
            draw.line((x, y, sx, sy), fill=color, width=width)
        elif draw_type == 1:
            draw.ellipse((x, y, sx, sy), fill=color, width=width)
        elif draw_type == 2:
            draw.ellipse((x, y, sx, sy), outline=color, width=width)
        elif draw_type == 3:
            draw.rectangle((x, y, sx, sy), fill=color, width=width)
        elif draw_type == 4:
            draw.rectangle((x, y, sx, sy), outline=color, width=width)
    return img


def render_image(kanji, font, fsize, angle):
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), WHITE)

    bg = render_clutter()
    img.alpha_composite(bg)
    jitter = 3

    img_k = render_kanji(kanji, font, fsize, angle)
    x = np.clip(IMG_SIZE//2 - img_k.width//2 + int(jitter*(random()*2-1)), 0, None)
    y = np.clip(IMG_SIZE//2 - img_k.height//2 + int(jitter*(random()*2-1)), 0, None)
    img.alpha_composite(img_k, (x, y))

    return img


def gen_training_sample(all_kanji, all_fonts):
    i_kanji = randint(0, len(all_kanji)-1)
    kanji = all_kanji[i_kanji]
    font = choice(all_fonts)
    fsize = randint(16, 24)  # randint(16, 180)
    angle = 0  # randint(0, 35)*10
    angle_noise = random()*10 - 5
    img = render_image(kanji, font, fsize, angle + angle_noise)
    return RawSample(np.array(img.convert("L"))/255, i_kanji, fsize, angle)
