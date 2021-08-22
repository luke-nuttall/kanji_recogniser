import tensorflow as tf
from tensorflow.keras import layers

from globals import IMG_SIZE, CATEGORIES_ANGLE, CATEGORIES_KANJI


class ResConv2D(layers.Layer):
    """
    Crude implementation of a ResNet unit with two convolutional layers.
    In this version the skip connection bypasses the activation function.
    UPDATE: Can now handle changes in filter size!
    """

    def __init__(self, filters: int, kernel_size: int, activation="relu", name=None):
        self.filters = filters
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.act1 = layers.Activation(activation)
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.act2 = layers.Activation(activation)
        self.diff_f = None

    def build(self, input_shape):
        # assume shape: [batch, y, x, filter]
        in_filters = input_shape[-1]
        self.diff_f = self.filters - in_filters

    def _add_skip_connection(self, inputs, x):
        if self.diff_f > 0:
            paddings = [[0, 0], [0, 0], [0, 0], [0, self.diff_f]]
            skip = tf.pad(inputs, paddings, "CONSTANT")
        elif self.diff_f < 0:
            skip = inputs[:, :, :, :self.filters]
        else:
            skip = inputs
        return x + skip

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self._add_skip_connection(inputs, x)
        return x


def build_regulariser() -> tf.keras.Model:
    l_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    l_next = layers.Conv2D(16, 3, activation="relu", name="conv1")(l_input)
    l_next = layers.Conv2D(64, 3, activation="relu", name="conv2")(l_next)
    l_next = layers.MaxPool2D()(l_next)
    l_next = layers.Flatten()(l_next)
    l_next = layers.Dense(64, activation="relu", name="dense1")(l_next)
    l_angle = layers.Dense(CATEGORIES_ANGLE, activation="softmax")(l_next)
    l_size = layers.Dense(1)(l_next)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_angle, l_size])
    return model


def build_recogniser(depth: int) -> tf.keras.Model:
    l_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    l_next = layers.Conv2D(32, 5, activation="relu", padding="same", name="conv1")(l_input)
    l_next = layers.MaxPool2D()(l_next)
    for ii in range(depth):
        l_next = ResConv2D(64, 3, name=f"resblock_{ii+1}")(l_next)
    l_next = layers.AveragePooling2D()(l_next)
    l_next = layers.Flatten()(l_next)
    l_next = layers.Dense(64, activation="relu", name="dense1")(l_next)
    l_kanji = layers.Dense(CATEGORIES_KANJI, activation="softmax")(l_next)
    model = tf.keras.Model(inputs=[l_input], outputs=[l_kanji], name="recogniser")
    return model
