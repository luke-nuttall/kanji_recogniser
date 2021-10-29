import datagen
from matplotlib import pyplot as plt


def main():
    help(datagen)
    arr = datagen.render_gl()
    plt.imshow(arr/255.0, cmap="gray", vmin=0, vmax=1)
    print(arr, type(arr))
    plt.show()


if __name__ == "__main__":
    main()
