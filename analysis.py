import json
from pathlib import Path
from matplotlib import pyplot as plt


def main():
    logs_nocur = Path("save") / "no_curriculum" / "training_log.json"
    logs_cur = Path("save") / "v1" / "training_log.json"

    fig, ax = plt.subplots(constrained_layout=True)
    ax: plt.Axes

    with logs_cur.open("r") as fp:
        data = json.load(fp)
        accuracy = [0] + [x['accuracy'] * x['n_kanji'] / 2500 for x in data]
        ax.plot(range(0, len(accuracy)), accuracy, label="Curriculum")

    with logs_nocur.open("r") as fp:
        data = json.load(fp)
        accuracy = [0] + [x['accuracy'] * x['n_kanji'] / 2500 for x in data]
        ax.plot(range(0, len(accuracy)), accuracy, label="No Curriculum")

    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color="black", alpha=0.25)

    ax.legend()
    ax.set_xlabel("Epoch (1 epoch = 32,000 samples)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 100)
    plt.show()


if __name__ == "__main__":
    main()
