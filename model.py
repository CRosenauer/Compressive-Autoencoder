import tensorflow as tf
import os
import glob
import cv2
import numpy


@tf.custom_gradient
def c_func(x):
    # needs to mirror Quantize as the model must function with float32 outputs due to the supported args of the loss
    # function. functionally it just simulates de-normalizing and clipping, but reverts the data to float os loss
    # measurement can be performed properly.
    result = tf.clip_by_value(x, 0.0, 1.0)
    result = result * 255.0
    result = tf.round(result)
    result = result / 255.0

    def grad(dy):
        return tf.ones_like(dy, dtype=tf.dtypes.float32)

    return result, grad


class ClipSimNormalize(tf.keras.layers.Layer):
    def __init__(self):
        super(ClipSimNormalize, self).__init__()

    def call(self, x):
        return c_func(x)


@tf.custom_gradient
def q_func(x):
    result = tf.clip_by_value(x, 0.0, 1.0)
    result = result * 255.0
    result = tf.round(result)
    result = result / 255.0

    def grad(dy):
        return dy

    return result, grad


class Quantize(tf.keras.layers.Layer):
    def __init__(self):
        super(Quantize, self).__init__()

    def call(self, x):
        return q_func(x)


def create_model():
    # encoder
    encoder_input = tf.keras.Input(shape=(512, 512, 3,), dtype=tf.float32, name="img_in")

    x = tf.pad(encoder_input, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)

    skip1 = x
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip1])

    skip2 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip2])

    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    encoder_output = tf.keras.layers.Conv2D(filters=96, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)

    # quantize
    decoder_input = Quantize()(encoder_output)

    # decoder
    sub_pixel_factor = 2

    # sub-pixel conv 1
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], input_shape=decoder_input.shape, padding="same")(decoder_input)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)
    skip3 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)

    x = tf.keras.layers.add([x, skip3])
    skip4 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.add([x, skip4])

    # sub-pixel conv 2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # sub-pixel conv 3
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # clip
    decoder_output = ClipSimNormalize()(x)

    autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name="CAE_model")
    return autoencoder


if __name__ == "__main__":
    CAE = create_model()
    CAE.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
            )

    # need to trip data. not enough ram
    X_data = []
    files = glob.glob("./data/130k/toys/*.jpeg", recursive=True)
    for myFile in files:
        image = cv2.imread(myFile)
        X_data.append(image)

    X_data = numpy.array(X_data)
    X_data.astype('float32') / 255

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # CAE.load_weights(checkpoint_path)

    CAE.fit(X_data, X_data, epochs=32, shuffle=True, callbacks=[cp_callback], validation_split=0.25)
    CAE.evaluate(X_data, X_data, callbacks=[cp_callback])
