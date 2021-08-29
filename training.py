import tensorflow as tf

from globals import CATEGORIES_KANJI, ALL_KANJI
from pipeline import Provider


class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.data = []
        self.extra_params = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, val in self.extra_params.items():
            logs[key] = val
        self.data.append(logs)


def train_simple(model: tf.keras.Model, provider: Provider):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    dataset = provider.get_dataset()
    dataset = dataset.map(lambda img, kanji, fsize, angle: (img, kanji))
    provider.kanji = ALL_KANJI

    logger = LoggerCallback()
    logger.extra_params['n_kanji'] = 2500

    print(f"Training on all {CATEGORIES_KANJI} kanji:")
    print(f"Training with a batch size of 16:")
    logger.extra_params['batch_size'] = 16
    model.fit(dataset.batch(16), steps_per_epoch=2000, epochs=25, callbacks=[logger])
    print(f"Training with a batch size of 32:")
    logger.extra_params['batch_size'] = 32
    model.fit(dataset.batch(32), steps_per_epoch=1000, epochs=25, callbacks=[logger])
    print(f"Training with a batch size of 64:")
    logger.extra_params['batch_size'] = 64
    model.fit(dataset.batch(64), steps_per_epoch=500, epochs=25, callbacks=[logger])
    print(f"Training with a batch size of 128:")
    logger.extra_params['batch_size'] = 128
    model.fit(dataset.batch(128), steps_per_epoch=250, epochs=25, callbacks=[logger])

    return logger.data


def train_curriculum(model: tf.keras.Model, provider: Provider):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    dataset = provider.get_dataset()
    dataset = dataset.map(lambda img, kanji, fsize, angle: (img, kanji))

    logger = LoggerCallback()
    logger.extra_params['batch_size'] = 16

    #last_write_counter = buf.write_counter

    for n_kanji in range(100, CATEGORIES_KANJI+1, 100):
        logger.extra_params['n_kanji'] = n_kanji
        print(f"Training on subset of {n_kanji} kanji:")
        provider.kanji = ALL_KANJI[:n_kanji]
        model.fit(dataset.batch(16), steps_per_epoch=2000, epochs=1, callbacks=[logger])
        #write_counter = buf.write_counter
        #print(f"There were {write_counter - last_write_counter} writes to the buffer during this epoch.")
        #last_write_counter = write_counter

    logger.extra_params['n_kanji'] = 2500

    print(f"Training on all {CATEGORIES_KANJI} kanji:")
    provider.kanji = ALL_KANJI
    print(f"Training with a batch size of 32:")
    logger.extra_params['batch_size'] = 32
    model.fit(dataset.batch(32), steps_per_epoch=1000, epochs=25, callbacks=[logger])
    print(f"Training with a batch size of 64:")
    logger.extra_params['batch_size'] = 64
    model.fit(dataset.batch(64), steps_per_epoch=500, epochs=25, callbacks=[logger])
    print(f"Training with a batch size of 128:")
    logger.extra_params['batch_size'] = 128
    model.fit(dataset.batch(128), steps_per_epoch=250, epochs=25, callbacks=[logger])

    return logger.data
