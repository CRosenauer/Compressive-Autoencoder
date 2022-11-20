import tensorflow as tf
import os
import glob
import cv2
import numpy
import math


@tf.custom_gradient
def c_func(x):
    result = tf.clip_by_value(x, 0.0, 1.0)

    def grad(dy):
        grad_out = tf.ones_like(dy, dtype=tf.dtypes.float32)
        return grad_out

    return result, grad


class ClipSimNormalize(tf.keras.layers.Layer):
    def __init__(self):
        super(ClipSimNormalize, self).__init__()
    def call(self, x):
        return c_func(x)


@tf.grad_pass_through
def q_func(x):
    result = tf.clip_by_value(x, 0.0, 1.0)
    result = result * 255.0
    result = tf.round(result)
    result = result / 255.0

    def grad(dy):
        grad_out = tf.ones_like(dy, dtype=tf.dtypes.float32)
        return grad_out

    return result, grad


class QuantizeEmulation(tf.keras.layers.Layer):
    def __init__(self):
        super(QuantizeEmulation, self).__init__()
    def call(self, x):
        # return q_func(x)
        return x

def r_func(x):
    result = x * 255.0
    result = tf.round(result)
    result = result / 255.0

    def grad(dy):
        grad_out = tf.ones_like(dy, dtype=tf.dtypes.float32)
        return grad_out

    return result, grad


class Round(tf.keras.layers.Layer):
    def __init__(self):
        super(Round, self).__init__()
    def call(self, x):
        return q_func(x)

def create_model():
    # encoder
    encoder_input = tf.keras.Input(shape=(128, 128, 3), dtype=tf.float32, name="img_in")

    x = tf.pad(encoder_input, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)

    skip1 = x
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip1])

    skip2 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip2])

    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)
    # x = tf.keras.activations.sigmoid(x)
    # encoder_output = Round()(x)
    encoder_output = x

    # quantize
    decoder_input = QuantizeEmulation()(encoder_output)
    # decoder_input = encoder_output

    # decoder
    sub_pixel_factor = 2

    # sub-pixel conv 1
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], input_shape=decoder_input.shape, padding="same")(decoder_input)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)
    skip3 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)

    x = tf.keras.layers.add([x, skip3])
    skip4 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.add([x, skip4])

    # sub-pixel conv 2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # sub-pixel conv 3
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # clip
    # decoder_output = ClipSimNormalize()(x)
    # decoder_output = tf.keras.activations.sigmoid(x)
    # decoder_output = Round()(x)
    decoder_output = QuantizeEmulation()(x)

    autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name="CAE_model")

    autoencoder.summary()

    return autoencoder


if __name__ == "__main__":
    CAE = create_model()
    CAE.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
            )

    files = glob.glob("./data/training/*.png", recursive=True)

    n_files = len(files)

    itr_size = 2000
    n_itrs = math.ceil(n_files / itr_size)

    for i in range(0, n_itrs):
        # need to trip data. not enough ram
        X_data = []

        for j in range(i * itr_size, min(n_files, (i + 1) * itr_size)):
            image = cv2.imread(files[j])
            X_data.append(image / 255.0)

        X_data = numpy.array(X_data, dtype='float32')

        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        # CAE.load_weights(checkpoint_path)

        CAE.fit(X_data, X_data, epochs=100000, shuffle=True, validation_split=0) # callbacks=[cp_callback],
        break

    model_input = tf.constant(X_data[0], dtype=tf.dtypes.float32, shape=[128, 128, 3])
    model_output = CAE(X_data)

    model_output = model_output * 255
    model_output = model_output[0].numpy()
    model_output = model_output.astype('uint8')

    cv2.imwrite("output.png", model_output)
    # im = PIL.Image.fromarray(model_output)
    # im.save("output.png")

    # CAE.evaluate(X_data, X_data)
