import json
from pathlib import Path

from matplotlib import pyplot as plt, ticker
import numpy as np

from globals import ALL_KANJI, ALL_FONTS, CATEGORIES_KANJI
from rendering import gen_training_sample


def plot_accuracy_comparison():
    save_dir = Path("save")
    saves = [(save_dir / "multiprocess", "Curriculum"),
             (save_dir / "no_curriculum", "No Curriculum")]

    fig, ax = plt.subplots(constrained_layout=True)
    ax: plt.Axes  # this type hint only exists to make autocompletion work better in PyCharm

    for folder, label in saves:
        logfile = folder / "training_log.json"
        with logfile.open("r") as fp:
            data = json.load(fp)
            # the line below is for correcting the raw accuracy for epochs where the model is only being trained
            # on a subset of the kanji.
            # If it achieves 50% accuracy on 50% of the kanji then its real accuracy is only 25%.
            # A zero is prepended so that the plot starts from the beginning of training rather than the end of epoch 1.
            accuracy = [0] + [x['accuracy'] * x['n_kanji'] / CATEGORIES_KANJI for x in data]
            ax.plot(range(len(accuracy)), accuracy, label=label)

    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color="black", alpha=0.25)

    ax.legend()
    ax.set_xlabel("Epoch (1 epoch = 32,000 samples)")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlim(0, 100)
    fig.savefig("plots/accuracy_comparison.png", dpi=150)
    plt.show()


def plot_speed_comparison():
    save_dir = Path("save")
    saves = [(save_dir / "multiprocess", "Multi-Process"),
             (save_dir / "multithread", "Multi-Threaded"),
             (save_dir / "singlethread", "Single-Threaded")]

    fig, ax = plt.subplots(constrained_layout=True)
    ax: plt.Axes

    for folder, label in saves:
        logfile = folder / "training_log.json"
        with logfile.open("r") as fp:
            data = json.load(fp)
            durations = [x['time_taken'] for x in data]
            ax.plot(np.arange(0.5, len(durations)), durations, label=label)

    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color="black", alpha=0.25)

    ax.legend()
    ax.set_xlabel("Epoch (1 epoch = 32,000 samples)")
    ax.set_ylabel("Time per epoch (s)")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    fig.savefig("plots/speed_comparison.png", dpi=150)
    plt.show()


def plot_gpu_comparison():
    save_dir = Path("save")
    saves = [(save_dir / "multiprocess", "Multi-Process"),
             (save_dir / "multithread", "Multi-Threaded"),
             (save_dir / "singlethread", "Single-Threaded")]

    fig, ax = plt.subplots(constrained_layout=True)
    ax: plt.Axes

    for folder, label in saves:
        logfile = folder / "training_log.json"
        with logfile.open("r") as fp:
            data = json.load(fp)
            gpu_util = [x['gpu_utilization'] for x in data]
            ax.plot(np.arange(0.5, len(gpu_util)), gpu_util, label=label)

    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color="black", alpha=0.25)

    ax.legend()
    ax.set_xlabel("Epoch (1 epoch = 32,000 samples)")
    ax.set_ylabel("Mean GPU Utilization (%)")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    fig.savefig("plots/gpu_util_comparison.png", dpi=150)
    plt.show()


def plot_sample_kanji():
    n_rows = 4
    n_cols = 8
    scale = 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale), constrained_layout=True)
    for ax in axes.flatten():
        ax: plt.Axes
        sample = gen_training_sample(ALL_KANJI, ALL_FONTS)
        ax.imshow(sample.image, cmap="gray", vmin=0, vmax=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    fig.savefig("plots/kanji_sample.png", dpi=150)
    plt.show()


def main():
    plot_accuracy_comparison()
    plot_speed_comparison()
    plot_gpu_comparison()
    plot_sample_kanji()


if __name__ == "__main__":
    main()
