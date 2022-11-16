import tensorflow as tf
import os
import random
import glob
import cv2
import numpy


lmbda = 3000


def pass_through_loss(_, x):
    return x


def pass_through_distortion_approx(_, x):
    return 1

def create_model():
    # encoder
    encoder_input = tf.keras.Input(shape=(512, 512, 3,), dtype=tf.float32, name="img_in")

    # normalization
    # x = tf.cast(encoder_input, dtype=tf.float32)
    # x = x / 255.0

    x = tf.pad(encoder_input, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], strides=2, input_shape=x.shape, activation='relu')(x)
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], strides=2, input_shape=x.shape, activation='relu')(x)

    skip1 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, activation='relu')(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, activation='relu')(x)

    x = tf.keras.layers.add([x, skip1])

    skip2 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, activation='relu')(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, activation='relu')(x)

    x = tf.keras.layers.add([x, skip2])

    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    encoder_output = tf.keras.layers.Conv2D(filters=96, kernel_size=[5, 5], strides=2, input_shape=x.shape,
                                            activation='relu')(x)

    # quantize

    # decoder
    sub_pixel_factor = 2

    # sub-pixel conv 1
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], input_shape=encoder_output.shape, padding="same",
                               activation='relu')(encoder_output)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)
    skip3 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)

    x = tf.keras.layers.add([x, skip3])
    skip4 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)
    x = tf.keras.layers.add([x, skip4])

    # sub-pixel conv 2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # sub-pixel conv 3
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(
        x)
    decoder_output = tf.nn.depth_to_space(x, sub_pixel_factor)

    # denormalize
    # x = x * 255.0

    # clip
    # x = tf.clip_by_value(x, 0.0, 255.0)
    # x = tf.math.floor(x)
    # decoder_output = tf.cast(x, dtype=tf.float32)

    autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name="CAE_model")
    return autoencoder


if __name__ == "__main__":
    CAE = create_model()
    CAE.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy']
            )

    data_dir = os.path.dirname("data/130k")
    batch_size = 1000

    # need to trip data. not enough ram
    X_data = []
    files = glob.glob("./data/130k/artwork/*.jpeg", recursive=True)
    for myFile in files:
        image = cv2.imread(myFile)
        X_data.append(image)

    X_data = numpy.array(X_data)
    X_data.astype('float32') / 255

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # CAE.load_weights(checkpoint_path)

    CAE.fit(X_data, X_data, epochs=32, shuffle=True, callbacks=[cp_callback])
    CAE.evaluate(X_data, X_data, callbacks=[cp_callback])
