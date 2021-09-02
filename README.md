# Kanji Recogniser

Experiments with using a ResNet-style deep convolutional neural network to perform OCR on noisy images of Japanese kanji.

This mainly exists as a learning exercise rather than anything useful.

## Requirements

* Python version 3.6 or later, with the following libraries:
  * TensorFlow
  * Numpy
  * Matplotlib
  * Pillow
* `nvidia-smi` somewhere on your PATH (should be installed automatically with nvidia GPU drivers)

## Usage

After downloading/cloning the repository you should run `get_fonts.sh` to download a basic set of Japanese fonts.
By defualt only the Noto series of fonts are downloaded, but other fonts can be added by putting them in the fonts folder.

See `python3 main.py --help` for a list of supported arguments.


## Summary of results so far

* Curriculum learning is an incredibly effective technique.
* Generating training data in a separate process (using the multiprocessing module) can give a big performance boost.

Look in the `plots` folder for some graphs showing the comparisons.