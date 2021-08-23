import time

from pathlib import Path

from matplotlib import pyplot as plt, font_manager
import numpy as np

import tensorflow as tf

from globals import ALL_FONTS, ALL_KANJI, IMG_SIZE, CATEGORIES_KANJI
from models import build_recogniser
from pipeline import ProviderMultiprocess

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

    provider = ProviderMultiprocess(ALL_KANJI[:100], ALL_FONTS)
    provider.start_background_tasks()
    time.sleep(1)
    dataset = provider.get_dataset()
    plot_sample_from_dataset(dataset)

    m_kanji = build_recogniser(10)
    m_kanji.summary()
    m_kanji.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_dir = Path("save") / "v3"
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
        last_time = time.time()

        for n_kanji in range(100, CATEGORIES_KANJI, 100):
            print(f"Training on subset of {n_kanji} kanji:")
            provider.kanji = ALL_KANJI[:n_kanji]
            #time.sleep(1)
            m_kanji.fit(dataset_kanji.batch(16), steps_per_epoch=2000, epochs=1)
            #write_counter = buf.write_counter
            #print(f"There were {write_counter - last_write_counter} writes to the buffer during this epoch.")
            #last_write_counter = write_counter
            print(f"Elapsed time: {time.time() - last_time:.2f} s")
            last_time = time.time()

        print(f"Training on all {CATEGORIES_KANJI} kanji:")
        provider.kanji = ALL_KANJI
        m_kanji.fit(dataset_kanji.batch(32), steps_per_epoch=1000, epochs=100)

        print("Stopping writer thread.")
        provider.stop_background_tasks()

        print(f"Saving model weights to: {save_path}")
        m_kanji.save_weights(str(save_path))
    else:
        provider.stop_background_tasks()
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

        ground_truth = ALL_KANJI[kanji]
        prediction = ALL_KANJI[np.argmax(pred[0])]
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
