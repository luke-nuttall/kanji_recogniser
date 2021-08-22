import threading
import time
from typing import List

from PIL import Image, ImageDraw, ImageFont
from random import choice, randint, random
from pathlib import Path
from collections import namedtuple

from matplotlib import pyplot as plt, font_manager
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

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


TrainingSample = namedtuple("TrainingSample", ["image", "kanji_index", "font_size", "angle"])


class CircularBuffer:
    def __init__(self, max_size: int):
        self.data: List[TrainingSample] = []
        self.capacity = max_size
        self.ptr_write = 0
        self.ptr_read = 0
        self.lock = threading.Lock()
        #self.write_counter = 0

    def add(self, value):
        with self.lock:
            if len(self.data) < self.capacity:
                self.data.append(value)
            else:
                self.data[self.ptr_write] = value
                self.ptr_write += 1
                self.ptr_write %= self.capacity
            #self.write_counter += 1

    def __iter__(self):
        return self

    def __next__(self) -> TrainingSample:
        with self.lock:
            val = self.data[self.ptr_read]
            self.ptr_read += 1
            self.ptr_read %= len(self.data)
            return val

    def get_generator(self):
        while True:
            yield next(self)


class WriterThread(threading.Thread):
    def __init__(self, buffer: CircularBuffer, kanji: list):
        super().__init__()
        self.buf = buffer
        self.__kanji = kanji
        self.halt = False
        self.lock = threading.Lock()
        self.__fonts = load_fonts()

    def run(self):
        print("Writer is running...")
        while not self.halt:
            with self.lock:
                val = gen_training_sample(self.__kanji, self.__fonts)
            self.buf.add(val)

    @property
    def kanji(self):
        with self.lock:
            return self.__kanji

    @kanji.setter
    def kanji(self, values):
        with self.lock:
            self.__kanji = values


class ReaderThread(threading.Thread):
    def __init__(self, buffer: CircularBuffer):
        super().__init__()
        self.buf = buffer
        self.halt = False

        plt.ion()
        self.fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        self.fig: plt.Figure
        self.images = []
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)
        img[0, 0] = 1.0
        for ax in axes.flatten():
            self.images.append(ax.imshow(img, cmap="gray"))

    def run(self):
        print("Reader is running...")
        while not self.halt:
            for image in self.images:
                image.set_data(next(self.buf).image)
            time.sleep(1.0)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


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
    return TrainingSample(
        tf.convert_to_tensor(np.array(img.convert("L"))/255),
        i_kanji,
        fsize,
        tf.one_hot(angle//10, CATEGORIES_ANGLE)
    )


def build_dataset(buf: CircularBuffer):
    return tf.data.Dataset.from_generator(buf.get_generator, output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
    ))


class ResConv2D(layers.Layer):
    """
    Crude implementation of a ResNet unit with two convolutional layers.
    In this version the skip connection bypasses the activation function.
    UPDATE: Can now handle changes in filter size!
    """

    def __init__(self, filters: int, kernel_size: int, activation="relu", name=None):
        self.filters = filters
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.act1 = layers.Activation(activation)
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.act2 = layers.Activation(activation)
        self.diff_f = None

    def build(self, input_shape):
        # assume shape: [batch, y, x, filter]
        in_filters = input_shape[-1]
        self.diff_f = self.filters - in_filters

    def _add_skip_connection(self, input, x):
        # return x + input
        if self.diff_f > 0:
            paddings = [[0, 0], [0, 0], [0, 0], [0, self.diff_f]]
            skip = tf.pad(input, paddings, "CONSTANT")
        elif self.diff_f < 0:
            skip = input[:, :, :, :self.filters]
        else:
            skip = input
        return x + skip

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self._add_skip_connection(inputs, x)
        return x


def build_regulariser():
    l_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    l_next = layers.Conv2D(16, 3, activation="relu", name="conv1")(l_input)
    l_next = layers.Conv2D(64, 3, activation="relu", name="conv2")(l_next)
    l_next = layers.MaxPool2D()(l_next)
    l_next = layers.Flatten()(l_next)
    l_next = layers.Dense(64, activation="relu", name="dense1")(l_next)
    l_angle = layers.Dense(CATEGORIES_ANGLE, activation="softmax")(l_next)
    l_size = layers.Dense(1)(l_next)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_angle])
    return model


def build_recogniser(depth: int):
    l_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    l_next = layers.Conv2D(32, 5, activation="relu", padding="same", name="conv1")(l_input)
    l_next = layers.MaxPool2D()(l_next)
    for ii in range(depth):
        l_next = ResConv2D(64, 3, name=f"resblock_{ii+1}")(l_next)
    l_next = layers.AveragePooling2D()(l_next)
    l_next = layers.Flatten()(l_next)
    l_next = layers.Dense(64, activation="relu", name="dense1")(l_next)
    l_kanji = layers.Dense(CATEGORIES_KANJI, activation="softmax")(l_next)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_kanji], name="recogniser")
    return model


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
    do_training = True

    all_kanji = list(load_kanji())
    buf = CircularBuffer(250)
    dataset = build_dataset(buf).prefetch(100)
    prod = WriterThread(buf, all_kanji[:100])
    prod.start()
    time.sleep(1)
    plot_sample_from_dataset(dataset)

    m_kanji = build_recogniser(10)
    m_kanji.summary()
    m_kanji.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_dir = Path("save") / "v2"
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

    if do_training:
        dataset_kanji = dataset.map(lambda img, kanji, fsize, angle: (img, kanji))
        #last_write_counter = buf.write_counter

        for n_kanji in range(100, CATEGORIES_KANJI, 100):
            print(f"Training on subset of {n_kanji} kanji:")
            prod.kanji = all_kanji[:n_kanji]
            time.sleep(1)
            m_kanji.fit(dataset_kanji.batch(8), steps_per_epoch=4000, epochs=1)
            #write_counter = buf.write_counter
            #print(f"There were {write_counter - last_write_counter} writes to the buffer during this epoch.")
            #last_write_counter = write_counter

        print(f"Training on all {CATEGORIES_KANJI} kanji:")
        prod.kanji = all_kanji
        m_kanji.fit(dataset_kanji.batch(32), steps_per_epoch=1000, epochs=100)

        print("Stopping writer thread.")
        prod.halt = True

        print(f"Saving model weights to: {save_path}")
        m_kanji.save_weights(str(save_path))
    else:
        prod.halt = True
        print(f"Loading model weights from: {save_path}")
        m_kanji.load_weights(str(save_path))

    '''
    This is for showing the performance of the kanji classifier.
    '''

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


if __name__ == "__main__":
    main()
