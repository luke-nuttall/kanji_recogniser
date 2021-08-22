import threading
import time
from typing import List

from pathlib import Path
from collections import namedtuple

from matplotlib import pyplot as plt, font_manager
import numpy as np

import tensorflow as tf

from models import build_recogniser
from rendering import gen_training_sample

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
                img, i_kanji, fsize, angle = gen_training_sample(self.__kanji, self.__fonts, IMG_SIZE)
            sample = TrainingSample(
                tf.convert_to_tensor(img),
                i_kanji,
                fsize,
                tf.one_hot(angle//10, CATEGORIES_ANGLE)
            )
            self.buf.add(sample)

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


def build_dataset(buf: CircularBuffer):
    return tf.data.Dataset.from_generator(buf.get_generator, output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
    ))


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

    m_kanji = build_recogniser(10, IMG_SIZE, CATEGORIES_KANJI)
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
