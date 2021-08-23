import multiprocessing
import threading
from typing import List
from collections import namedtuple
import tensorflow as tf

from rendering import gen_training_sample, RawSample
from globals import IMG_SIZE, CATEGORIES_ANGLE

TFSample = namedtuple("TFSample", ["image", "kanji_index", "font_size", "angle"])


def dataset_from_iterator(iterator):
    def gen():
        for value in iterator:
            yield value
    return tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
    ))


def convert_sample(raw: RawSample) -> TFSample:
    return TFSample(
        tf.convert_to_tensor(raw.image),
        raw.kanji_index,
        raw.font_size,
        tf.one_hot(int(CATEGORIES_ANGLE * raw.angle/360), CATEGORIES_ANGLE)
    )


class CircularBuffer:
    """
    A thread-safe circular buffer implemented on top of a Python list.
    All operations are O(1).
    Items are not removed when they are read.

    The idea is that the buffer will keep returning values regardless of any speed difference between the thread
    writing to the buffer and the one reading from it.
    If the writing thread is slower than the reading thread then the same item may be returned multiple times.
    """
    def __init__(self, max_size: int):
        self.__data: List[TFSample] = []
        self.capacity = max_size
        self.ptr_write = 0
        self.ptr_read = 0
        self.lock = threading.Lock()
        #self.write_counter = 0

    def add(self, value):
        with self.lock:
            if len(self.__data) < self.capacity:
                self.__data.append(value)
            else:
                self.__data[self.ptr_write] = value
                self.ptr_write += 1
                self.ptr_write %= self.capacity
            #self.write_counter += 1

    def __iter__(self):
        return self

    def __next__(self) -> TFSample:
        with self.lock:
            val = self.__data[self.ptr_read]
            self.ptr_read += 1
            self.ptr_read %= len(self.__data)
            return val

    def get_generator(self):
        while True:
            yield next(self)


class MT_Thread(threading.Thread):
    def __init__(self, buffer: CircularBuffer, kanji: list, fonts: list):
        super().__init__()
        self.buf = buffer
        self.__kanji = kanji
        self.halt = False
        self.lock = threading.Lock()
        self.__fonts = fonts

    def run(self):
        print("Writer is running...")
        while not self.halt:
            with self.lock:
                raw = gen_training_sample(self.__kanji, self.__fonts)
            sample = convert_sample(raw)
            self.buf.add(sample)

    @property
    def kanji(self):
        with self.lock:
            return self.__kanji

    @kanji.setter
    def kanji(self, values):
        with self.lock:
            self.__kanji = values


class MP_WriterProcess(multiprocessing.Process):
    def __init__(self, kanji: list, fonts: list):
        super().__init__()
        self.__kanji = kanji
        self.halt = multiprocessing.Value('b', False)  # shared memory
        self.__fonts = fonts
        self.conn_parent, self.conn_child = multiprocessing.Pipe()

    def run(self):
        print("Writer Process is running...")
        while not self.halt.value:
            while self.conn_child.poll():
                self.__kanji = self.conn_child.recv()
            val = gen_training_sample(self.__kanji, self.__fonts)
            self.conn_child.send(val)
        print("Writer Process has stopped")

    def set_kanji(self, values):
        self.conn_parent.send(values)


class MP_WriterThread(threading.Thread):
    def __init__(self, buffer: CircularBuffer, conn_parent):
        super().__init__()
        self.buf = buffer
        self.halt = False
        self.conn_parent = conn_parent

    def run(self):
        print("Writer Thread is running...")
        while not self.halt:
            raw_sample = self.conn_parent.recv()
            sample = convert_sample(raw_sample)
            self.buf.add(sample)
        print("Writer Thread has stopped")


class Provider:
    """
    This class provides a standard interface to various systems for generating and buffering training data.
    The base class just implements a simple single-threaded data generation pipeline.
    See the other subclasses for faster algorithms.
    """
    def __init__(self, kanji: list, fonts: list):
        self.__kanji = kanji
        self.__fonts = fonts

    def start_background_tasks(self):
        pass

    def stop_background_tasks(self):
        pass

    @property
    def kanji(self):
        return self.__kanji

    @kanji.setter
    def kanji(self, values):
        self.__kanji = values

    def __generator(self):
        while True:
            raw = gen_training_sample(self.__kanji, self.__fonts)
            yield convert_sample(raw)

    def get_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(self.__generator, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
        ))


class ProviderMultithread(Provider):
    def __init__(self, kanji: list, fonts: list):
        super().__init__(kanji, fonts)
        self.buf = CircularBuffer(100)
        self.thread = MT_Thread(self.buf, kanji, fonts)

    def start_background_tasks(self):
        self.thread.start()

    def stop_background_tasks(self):
        self.thread.halt = True

    @property
    def kanji(self):
        return self.__kanji

    @kanji.setter
    def kanji(self, values):
        self.__kanji = values
        self.thread.kanji = values

    def __generator(self):
        for sample in self.buf:
            yield sample

    def get_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(self.__generator, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
        ))


class ProviderMultiprocess(Provider):
    def __init__(self, kanji: list, fonts: list):
        super().__init__(kanji, fonts)
        self.buf = CircularBuffer(100)
        self.proc = MP_WriterProcess(kanji, fonts)
        self.thread = MP_WriterThread(self.buf, self.proc.conn_parent)

    def start_background_tasks(self):
        self.proc.start()
        self.thread.start()

    def stop_background_tasks(self):
        self.proc.halt.value = True
        self.thread.halt = True

    @property
    def kanji(self):
        return self.__kanji

    @kanji.setter
    def kanji(self, values):
        self.__kanji = values
        self.proc.set_kanji(values)

    def __generator(self):
        for sample in self.buf:
            yield sample

    def get_dataset(self) -> tf.data.Dataset:
        # This is an exact copy of the same method from the superclass
        # If the world were a sane, reasonable place it would be completely pointless to put this here
        # However, the world is evidently not a sane, reasonable place.
        # Adding this duplicate code speeds up execution by a factor of about 3.
        # I think gnomes are to blame.
        return tf.data.Dataset.from_generator(self.__generator, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(), dtype=tf.int16),
            tf.TensorSpec(shape=(CATEGORIES_ANGLE,), dtype=tf.int16),
        ))
        # More seriously, I would guess that it's a difference in how the "self.__generator" lookup is performed
        # Maybe in one version the lookup is only performed once, while in the other it happens every time a
        # value is requested from the dataset.
        # This would be similar to the concept of virtual methods in low-level languages.
