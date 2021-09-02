import json
import time
from pathlib import Path
import argparse

from matplotlib import pyplot as plt, font_manager
import numpy as np

import tensorflow as tf

from globals import ALL_FONTS, ALL_KANJI, IMG_SIZE
from models import build_recogniser
from pipeline import Provider, ProviderMultithread, ProviderMultiprocess
from training import train_curriculum, train_simple

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
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--notrain", action="store_true",
                       help="No training will occur. The model will be loaded from the last saved copy.")
    group.add_argument("-t", "--train", choices=["simple", "thread", "process"], default="process",
                       help="Controls the mode used to train the mode. If not specified the fastest mode will be used.")

    parser.add_argument("-c", "--curriculum", action="store_true",
                        help="Use curriculum learning. The model will start by learning the first 100 kanji, then "
                             "subsequent epochs will slowly introduce more kanji.")
    args = parser.parse_args()

    if args.notrain:
        provider = Provider(ALL_KANJI, ALL_FONTS)
        save_dir = Path("save") / "multiprocess"
    else:
        if args.train == "simple":
            provider = Provider(ALL_KANJI[:100], ALL_FONTS)
            save_dir = Path("save") / "singlethread"
        elif args.train == "thread":
            provider = ProviderMultithread(ALL_KANJI[:100], ALL_FONTS)
            save_dir = Path("save") / "multithread"
        elif args.train == "process":
            provider = ProviderMultiprocess(ALL_KANJI[:100], ALL_FONTS)
            save_dir = Path("save") / "multiprocess"
        else:
            provider = ProviderMultiprocess(ALL_KANJI[:100], ALL_FONTS)
            save_dir = Path("save") / "multiprocess"

    provider.start_background_tasks()
    time.sleep(1)

    m_kanji = build_recogniser(10)
    m_kanji.summary()

    if not args.curriculum:
        save_dir = Path("save") / "no_curriculum"
    save_path = save_dir / "recogniser_clutter"
    log_path = save_dir / "training_log.json"

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
    if args.notrain:
        print(f"Loading model weights from: {save_path}")
        m_kanji.load_weights(str(save_path))
    else:
        if args.curriculum:
            train_log = train_curriculum(m_kanji, provider)
        else:
            train_log = train_simple(m_kanji, provider)
        print(f"Saving model weights to: {save_path}")
        m_kanji.save_weights(str(save_path))
        print("\nLogs:")
        for epoch, row in enumerate(train_log):
            print(f"Epoch {epoch}: {row}")
        with log_path.open("w") as fp:
            json.dump(train_log, fp)

    """
    This is for showing the performance of the kanji classifier.
    """
    dataset = provider.get_dataset()
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
    prediction = ALL_KANJI[np.argmax(pred[0])]
    ground_truth = ALL_KANJI[kanji]
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

    provider.stop_background_tasks()


if __name__ == "__main__":
    main()
