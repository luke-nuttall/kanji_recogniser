import multiprocessing as mp
import threading
from collections import namedtuple
import queue  # for queue.Full and queue.Empty exceptions used by mp.Queue
from math import pi
from random import choice, randint, random

import tensorflow as tf

from rendering import gen_training_sample, TrainingSample
from globals import IMG_SIZE, ALL_KANJI, KANJI_INDICES, CATEGORIES_ANGLE
import datagen


def get_dataset_from_generator(gen):
    return tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
    ))


class GenThread(threading.Thread):
    def __init__(self, queue: mp.Queue, kanji: list, fonts: list, batch_size=16):
        super().__init__()
        self.queue = queue
        self.__kanji = kanji
        self.__halt = False
        self.lock = threading.Lock()
        self.__fonts = fonts
        self.batch_size = batch_size

    def run(self):
        print(f"{self.__class__.__name__} has started.")
        while not self.__halt:
            with self.lock:
                batch = [gen_training_sample(self.__kanji, self.__fonts) for _ in range(self.batch_size)]
            self.queue.put(batch)
        print(f"{self.__class__.__name__} has stopped.")

    def set_kanji(self, values):
        with self.lock:
            self.__kanji = values

    def halt(self):
        self.__halt = True


class GenProcess(mp.Process):
    def __init__(self, queue: mp.Queue, kanji: list, fonts: list, batch_size=16):
        super().__init__()
        self.__kanji = kanji
        self.__halt = mp.Value('b', False)  # shared memory
        self.__fonts = fonts
        self.conn_parent, self.conn_child = mp.Pipe()
        self.queue = queue
        self.__batch_size = batch_size
        # Batching is done to check for, and circumvent, any overhead
        # from rapidly enqueuing and dequeuing lots of small objects.

    def run(self):
        print(f"{self.__class__.__name__} has started.")
        while not self.__halt.value:
            while self.conn_child.poll():
                self.__kanji = self.conn_child.recv()
            batch = [gen_training_sample(self.__kanji, self.__fonts) for _ in range(self.__batch_size)]
            self.queue.put(batch)
        print(f"{self.__class__.__name__} has stopped.")

    def set_kanji(self, values):
        self.conn_parent.send(values)

    def halt(self):
        self.__halt.value = True


class Provider:
    """
    This class provides a standard interface to various systems for generating and buffering training data.
    The base class just implements a simple single-threaded data generation pipeline.
    See the other subclasses for faster algorithms.
    """
    def __init__(self, kanji: list, fonts: list):
        self._kanji = kanji
        self._fonts = fonts

    def start_background_tasks(self):
        """
        This should be called before attempting to use get_dataset().
        """
        pass

    def stop_background_tasks(self):
        """
        Irreversibly halt any background tasks.
        After calling this the dataset may stop producing output.
        """
        pass

    @property
    def kanji(self):
        return self._kanji

    @kanji.setter
    def kanji(self, values):
        self._kanji = values

    def __generator(self):
        while True:
            yield gen_training_sample(self._kanji, self._fonts)

    def get_dataset(self) -> tf.data.Dataset:
        return get_dataset_from_generator(self.__generator)


class ProviderRust(Provider):
    def __init__(self, kanji: list, fonts: list):
        super().__init__(kanji, fonts)
        self.renderer = datagen.RustRenderer(IMG_SIZE, [str(x) for x in fonts], ALL_KANJI, CATEGORIES_ANGLE)

    @property
    def kanji(self):
        return self.renderer.kanji

    @kanji.setter
    def kanji(self, values):
        self.renderer.kanji = values
        # print("Set kanji to:", "".join(self.renderer.kanji))

    def __generator(self):
        while True:
            for sample in self.renderer.render_batch(16):
                yield TrainingSample(*sample)

    def get_dataset(self) -> tf.data.Dataset:
        return get_dataset_from_generator(self.__generator)


class ProviderMultiBase(Provider):
    def __init__(self, kanji: list, fonts: list, n_threads=4):
        super().__init__(kanji, fonts)
        self.queue_size = 32
        self.manager = mp.Manager()  # see https://stackoverflow.com/a/46041587
        self.queue = self.manager.Queue(self.queue_size)
        self.threads = self.create_threads(kanji, fonts, n_threads)

    def create_threads(self, kanji, fonts, n_threads: int) -> list:
        """
        Subclasses should override this to return a new list of thread objects
        """
        return []

    def start_background_tasks(self):
        for thread in self.threads:
            thread.start()

    def stop_background_tasks(self):
        for thread in self.threads:
            thread.halt()
        self.__flush_queue()  # This just makes sure that the threads aren't stuck waiting for space in the queue.
        for thread in self.threads:
            thread.join()

    def __flush_queue(self):
        """
        Remove every item which is in the queue when this function is called.
        Items which are added to the queue while this function is running may or may not be removed.
        """
        for ii in range(self.queue_size):
            try:
                self.queue.get(block=False)
            except queue.Empty:
                return

    @property
    def kanji(self):
        return self.__kanji

    @kanji.setter
    def kanji(self, values):
        self.__kanji = values
        for thread in self.threads:
            thread.set_kanji(values)

    def __generator(self):
        while True:
            batch = self.queue.get(timeout=5)
            for sample in batch:
                yield sample

    def get_dataset(self) -> tf.data.Dataset:
        return get_dataset_from_generator(self.__generator)
        # This is an exact copy of the same method from the superclass
        # If the world were a sane, reasonable place it would be completely pointless to put this here
        # However, the world is evidently not a sane, reasonable place.
        # Adding this duplicate code speeds up execution by a factor of about 3.
        # I think gnomes are to blame.
        # More seriously, I would guess that it's a difference in how the "self.__generator" call is performed
        # Maybe the code path to call a method which has been overridden in a child class is much more complicated
        # than the code path when calling a method which is defined in the same class.
        # It might be similar to the concept of virtual methods in low-level languages.'''


class ProviderMultithread(ProviderMultiBase):
    def create_threads(self, kanji, fonts, n_threads: int):
        return [GenThread(self.queue, kanji, fonts) for _ in range(n_threads)]


class ProviderMultiprocess(ProviderMultiBase):
    def create_threads(self, kanji, fonts, n_threads: int):
        return [GenProcess(self.queue, kanji, fonts) for _ in range(n_threads)]