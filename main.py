from PIL import Image, ImageDraw, ImageFont
from random import choice, randint, random
from pathlib import Path

from matplotlib import pyplot as plt, font_manager
import numpy as np

import tensorflow as tf

from models import build_recogniser

font_manager.fontManager.addfont("fonts/NotoSansJP-Regular.otf")
plt.rc('font', family='Noto Sans JP')

"""
The GPU has a limited amount of memory.
Some of that memory will be allocated to the model, while some must be left free to allow essential libraries to be 
loaded onto the GPU.
Annoyingly Tensorflow allocates the memory for the model before it loads the libraries.
If too much memory gets allocated to the model then there won't be enough space left for libraries and we'll get 
confusing error messages about various libraries being unavailable even though they're installed in the OS.
Even more annoyingly TensorFlow has a nasty habit of allocating too much memory for the model.
The code below lets us specify exactly how much GPU memory to use for the model.
If the value is too small we'll at least get helpful error messages about "OUT OF MEMORY" rather than confusing ones
about libraries being unavailable.
"""
gpu_memory = 500
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)]
    )
    print(f"Found GPU: {gpus[0].name}")
    print(f"Setting GPU memory limit to {gpu_memory} MB")
else:
    print("No GPU found. Running on CPU. This may be very, very slow.")


def load_kanji():
    with open("kanji.txt", "r") as fp:
        return fp.read().strip()


def load_fonts():
    return list(Path("fonts").glob("*.otf"))


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMG_SIZE = 32

CATEGORIES_ANGLE = 36
CATEGORIES_KANJI = len(load_kanji())


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
    return img.convert("L"), i_kanji, fsize, angle


def dataset_gen(n_kanji=0):
    all_kanji = load_kanji()
    all_fonts = load_fonts()
    if n_kanji > 0:
        all_kanji = all_kanji[:n_kanji]
    while True:
        img, i_kanji, fsize, angle = gen_training_sample(all_kanji, all_fonts)
        image = tf.convert_to_tensor(np.array(img)/255)
        angle = tf.one_hot(angle//10, CATEGORIES_ANGLE)  # angle //= 10
        yield image, i_kanji, fsize, angle


def build_dataset(n_kanji=0):
    dataset = tf.data.Dataset.from_generator(dataset_gen,
                                             (tf.float32, tf.int16, tf.int16, tf.int16),
                                             (tf.TensorShape([IMG_SIZE, IMG_SIZE]), tf.TensorShape([]),
                                              tf.TensorShape([]), tf.TensorShape([CATEGORIES_ANGLE])),
                                             args=[n_kanji])
    dataset = dataset
    return dataset


def plot_sample_from_dataset(dataset):
    fig = plt.figure(figsize=(8, 8))
    for ii, row in enumerate(dataset.take(16)):
        ax: plt.Axes = fig.add_subplot(4, 4, ii+1)
        img = row[0].numpy()
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        size = row[2].numpy()
        angle = row[3].numpy().argmax() * 10
        ax.set_title(f"s:{size}, θ:{angle}")
    plt.tight_layout()
    plt.show()


def main():
    dataset = build_dataset().prefetch(100)
    plot_sample_from_dataset(dataset)

    m_kanji = build_recogniser(10, IMG_SIZE, CATEGORIES_KANJI)
    m_kanji.summary()
    m_kanji.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_dir = Path("save") / "v1"
    save_path = save_dir / "recogniser_clutter"

    """
    The code below is for training the network.
    Two techniques are used to improve training speed:
    1. New kanji are only introduced slowly. This is based off how humans learn best. It's easier to learn a few 
       examples first, then use that knowledge as a starting point to learn more. Trying to learn all 2500 classes 
       from the start produces very poor gradient signals, especially with a small minibatch size.
    2. The minibatch size starts off small but is later increased. With small sizes it gets lots of weight updates 
       quickly, but those updates have higher variance. With larger batches it gets fewer updates but they are 
       generally more accurate.
    """

    for n_kanji in range(100, CATEGORIES_KANJI, 100):
        print(f"Training on subset of {n_kanji} kanji:")
        ds_kanji = build_dataset(n_kanji).map(lambda img, kanji, fsize, angle: (img, kanji))
        m_kanji.fit(ds_kanji.batch(8), steps_per_epoch=4000, epochs=1)

    print(f"Training on all {CATEGORIES_KANJI} kanji:")
    ds_kanji = dataset.map(lambda img, kanji, fsize, angle: (img, kanji))
    m_kanji.fit(ds_kanji.batch(32), steps_per_epoch=2000, epochs=10)

    print(f"Saving model weights to: {save_path}")
    m_kanji.save_weights(str(save_path))

    '''
    This is for showing the performance of the kanji classifier.
    '''

    all_kanji = load_kanji()
    n_rows = 4
    n_cols = 8
    scale = 2
    fig = plt.figure(figsize=(n_cols * scale, n_rows * scale))
    for ii, (img, kanji, fsize, angle) in enumerate(dataset.take(n_rows * n_cols)):
        pred = m_kanji.predict(tf.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1)))

        ground_truth = all_kanji[kanji]
        prediction = all_kanji[np.argmax(pred[0])]
        confidence = max(pred[0])

        ax: plt.Axes = fig.add_subplot(n_rows, n_cols, ii + 1)
        ax.imshow(img.numpy(), cmap="gray", extent=(-1, 1, -1, 1), vmax=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

        red = np.array([1.0, 0.0, 0.0])
        green = np.array([0.0, 0.6, 0.0])
        color = red
        if ground_truth == prediction:
            color = green

        ax.set_title(f"{ground_truth} -> {prediction} ({int(confidence * 100)}%)", color=color)

    plt.tight_layout()
    plt.show()

    """
    The code below is for inspecting the inner workings of the network.
    """

    conv1 = m_kanji.get_layer("conv1")
    kernels = conv1.weights[0]
    kernels = np.transpose(kernels.numpy()[:, :, 0, :], axes=[2, 0, 1])

    n_total = len(kernels)
    n_cols = 16
    n_rows = int(np.ceil(n_total / n_cols))
    scale = 1

    fig1 = plt.figure(figsize=(n_cols * scale, n_rows * scale))

    for ii, k in enumerate(kernels):
        ax: plt.Axes = fig1.add_subplot(n_rows, n_cols, ii + 1)
        ax.imshow(k, cmap="gray", extent=(-1, 1, -1, 1))
        ax.set_xticks([])
        ax.set_yticks([])

    img, kanji, fsize, angle = next(dataset.as_numpy_iterator())

    fig2 = plt.figure(figsize=(4, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.imshow(img, cmap="gray", extent=(-1, 1, -1, 1))
    ax2.set_xticks([])
    ax2.set_yticks([])

    def functor_get_hidden_output(model, name: str):
        i_in = 0
        i_out = model.layers.index(model.get_layer(name))
        func = tf.keras.backend.function([model.layers[i_in].input],
                                         [model.layers[i_out].output])
        return func

    inpt = tf.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1))
    hidden = functor_get_hidden_output(m_kanji, "resblock_10")([inpt])[0]
    activations = np.transpose(hidden[0], axes=[2, 0, 1])

    pred = m_kanji.predict(inpt)
    prediction = all_kanji[np.argmax(pred[0])]
    ground_truth = all_kanji[kanji]
    print(f"Prediction: {prediction} ({prediction == ground_truth})")

    n_total = len(activations)
    n_cols = 16
    n_rows = int(np.ceil(n_total / n_cols))
    scale = 1

    fig3 = plt.figure(figsize=(n_cols * scale, n_rows * scale))

    for ii, k in enumerate(activations):
        ax: plt.Axes = fig3.add_subplot(n_rows, n_cols, ii + 1)
        ax.imshow(k, cmap="gray", extent=(-1, 1, -1, 1))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
