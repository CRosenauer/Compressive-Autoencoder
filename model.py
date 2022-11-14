import tensorflow as tf


def CreateModel():
    # encoder
    encoder_input = tf.keras.Input(shape=(512, 512, 3,), name="img_in")

    # normalization
    x = encoder_input / 255.0

    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)

    skip1 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip1])

    skip2 = x

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape)(x)

    x = tf.keras.layers.add([x, skip2])

    x = tf.pad(x, [[0, 0], [2, 2],[2, 2], [0, 0]], 'REFLECT')
    encoder_output = tf.keras.layers.Conv2D(filters=96, kernel_size=[5, 5], strides=2, input_shape=x.shape)(x)

    # round encoder output. ig
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()

    # decoder
    sub_pixel_factor = 2

    # sub-pixel conv 1
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], input_shape=encoder_output.shape, padding="same")(encoder_output)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)
    skip3 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)

    print(x.shape)
    print(skip3.shape)

    x = tf.keras.layers.add([x, skip3])
    skip4 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.keras.layers.add([x, skip4])

    # sub-pixel conv 2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # sub-pixel conv 3
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[3, 3], input_shape=x.shape, padding="same")(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # denormalize
    x = x * 255.0

    # clip
    # ignores gradient = 1 for now
    decoder_output = tf.clip_by_value(x, 0.0, 255.0);

    autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output, name="CAE_model")

    autoencoder.summary()

    return autoencoder

def MakeInputFn(inputData, numEpochs=32, shuffle=True, batchSize=32):
    def InputFunction():
        dataSet = tf.data.Dataset.from_tensor_slices(inputData, inputData)
        if shuffle:
            dataSet = dataSet.shuffle()
        dataSet = dataSet.batch(batchSize).repeat(numEpochs)
        return dataSet
    return InputFunction


def Train():
    print("Stub - Training")


def Evaluate():
    print("Stub - Evaluate")

CAE = CreateModel()
