# Kanji Recogniser

Experiments with using a ResNet-style deep convolutional neural network to perform OCR on noisy images of Japanese kanji.

This mainly exists as a learning exercise rather than anything terribly useful.

## Requirements

* Python version 3.6 or later
* TensorFlow
* Numpy
* Matplotlib
* Pillow

## Usage

See `python3 main.py --help` for a list of supported arguments.


## Summary of results so far

* Curriculum learning is an incredibly effective technique.
* Generating training data in a separate process (using the multiprocessing module) can give a big performance boost.