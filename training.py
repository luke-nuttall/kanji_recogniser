import time
import subprocess

import tensorflow as tf
import numpy as np

from globals import CATEGORIES_KANJI, ALL_KANJI
from pipeline import Provider


class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.data = []
        self.extra_params = {}
        self.last_time = None  # used to track how long each epoch takes to run
        self.proc = None

    def on_epoch_begin(self, epoch: int, logs=None):
        self.last_time = time.time()
        self.proc = subprocess.Popen(
            args=["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "--loop=1"],
            stdout=subprocess.PIPE, text=True
        )

    def on_epoch_end(self, epoch: int, logs=None):
        for key, val in self.extra_params.items():
            logs[key] = val
        logs['time_taken'] = time.time() - self.last_time

        self.proc.terminate()
        stdout, _ = self.proc.communicate()
        logs['gpu_utilization'] = np.mean([float(x) for x in stdout.split()])

        self.data.append(logs)


# A schedule is a list of phases
# Each phase is a dict containing one or more of the following fields:
#   n_epochs: int number of epochs in this phase (required!)
#   n_batches: int number of batches per epoch
#   batch_size: int number of samples per batch
#   learning_rate: float
#   kanji: list[char] - the set of kanji to train the network on

def train_schedule(model: tf.keras.Model, provider: Provider, schedule: list):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    dataset = provider.get_dataset()
    dataset = dataset.map(lambda img, kanji, fsize, angle: (img, kanji))

    logger = LoggerCallback()
    callbacks = [logger]

    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, profile_batch='500,520')
    # callbacks.append(tensorboard)

    # this stores the default parameters to be used for each phase of training
    # it is expected that these will be overwritten by values provided in the schedule
    state = {
        "n_epochs": 0,
        "n_batches": 2000,
        "batch_size": 16,
        "learning_rate": 0.001,
        "kanji": ALL_KANJI
    }

    total_epochs = sum(phase['n_epochs'] for phase in schedule)

    # iterate through the phases in the schedule, updating the current state with each new phase
    epoch = 1
    for phase in schedule:
        state.update(phase)
        logger.extra_params['n_batches'] = state['n_batches']
        logger.extra_params['batch_size'] = state['batch_size']
        logger.extra_params['learning_rate'] = state['learning_rate']
        logger.extra_params['n_kanji'] = len(state['kanji'])
        provider.kanji = state['kanji']
        model.optimizer.learning_rate.assign(state['learning_rate'])
        # iterate through the epochs in each phase
        for _ in range(state['n_epochs']):
            meta = "    ".join(f"{k}={v}" for k, v in logger.extra_params.items())
            print(f"Epoch {epoch}/{total_epochs}    {meta}")
            model.fit(dataset.batch(state['batch_size']),
                      epochs=1,
                      steps_per_epoch=state['n_batches'],
                      callbacks=callbacks)
            epoch += 1

    return logger.data


def train_simple(model: tf.keras.Model, provider: Provider):
    schedule = [
        {"n_epochs": 25, "batch_size": 16, "n_batches": 2000, "learning_rate": 0.001, "kanji": ALL_KANJI},
        {"n_epochs": 25, "batch_size": 32, "n_batches": 1000},
        {"n_epochs": 25, "batch_size": 64, "n_batches": 500},
        {"n_epochs": 25, "batch_size": 128, "n_batches": 250},
    ]
    return train_schedule(model, provider, schedule)


def train_curriculum(model: tf.keras.Model, provider: Provider, shuffle=False):
    # Make a function-local copy of the kanji list so we can shuffle it safely
    all_kanji = ALL_KANJI.copy()
    if shuffle:
        np.random.shuffle(all_kanji)

    schedule = []
    for n_kanji in range(100, CATEGORIES_KANJI + 1, 100):
        schedule.append({
            "n_epochs": 1,
            "batch_size": 16,
            "n_batches": 2000,
            "learning_rate": 0.001,
            "kanji": all_kanji[:n_kanji]
        })
    schedule.append({"n_epochs": 25, "batch_size": 32, "n_batches": 1000})
    schedule.append({"n_epochs": 25, "batch_size": 64, "n_batches": 500})
    schedule.append({"n_epochs": 25, "batch_size": 128, "n_batches": 250})

    return train_schedule(model, provider, schedule)
