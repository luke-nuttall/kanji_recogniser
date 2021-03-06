# Kanji Recogniser

Experiments with using a ResNet-style deep convolutional neural network to perform OCR on noisy images of Japanese kanji.

This mainly exists as a learning exercise rather than anything useful.

For a more detailed write-up please take a look at https://hackmd.io/@luke-nuttall/kanji-ocr

## Requirements

* Python version 3.6 or later, with the following libraries:
  * TensorFlow
  * Numpy
  * Matplotlib
  * Pillow
* `nvidia-smi` somewhere on your PATH (should be installed automatically with nvidia GPU drivers)

To make use of the rust code in this branch you should install the rust toolchain from https://rustup.rs/

In addition to the standard rust toolchain it's helpful to also create a python virtualenv and `pip install maturin`.
[Maturin](https://pypi.org/project/maturin/) works with cargo and pyo3 to make it very easy to build and install rust python modules with a single command.

## Usage

After downloading/cloning the repository you should run `get_fonts.sh` to download a basic set of Japanese fonts.
By defualt only the Noto series of fonts are downloaded, but other fonts can be added by putting them in the fonts folder.

See `python3 main.py --help` for a list of supported arguments.

It is recommended that you set the environment variable `TF_XLA_FLAGS=--tf_xla_auto_jit=2` when training the network.
Depending on your GPU it may give significant performance improvements.


## Summary of results so far

* Curriculum learning is an incredibly effective technique.
* Generating training data in a separate process (using the multiprocessing module) can give a big performance boost.

Look in the `plots` folder for some graphs showing the comparisons.