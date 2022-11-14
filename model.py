import tensorflow as tf
import tensorflow_compression as tfc
import os
import random


lmbda = 3000


def pass_through_loss(_, x):
    return x

def CreateModel():
    # encoder
    encoder_input = tf.keras.Input(shape=(512, 512, 3,), name="img_in")

    # normalization
    x = encoder_input / 255.0

    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
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

    x = tf.pad(x, [[0, 0], [2, 2],[2, 2], [0, 0]], 'REFLECT')
    encoder_output = tf.keras.layers.Conv2D(filters=96, kernel_size=[5, 5], strides=2, input_shape=x.shape, activation='relu')(x)

    # round encoder output. ig
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

    # decoder
    sub_pixel_factor = 2

    # sub-pixel conv 1
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], input_shape=encoder_output.shape, padding="same", activation='relu')(encoder_output)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)
    skip3 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)

    x = tf.keras.layers.add([x, skip3])
    skip4 = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)
    x = tf.keras.layers.add([x, skip4])

    # sub-pixel conv 2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # sub-pixel conv 3
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[3, 3], input_shape=x.shape, padding="same", activation='relu')(x)
    x = tf.nn.depth_to_space(x, sub_pixel_factor)

    # denormalize
    x = x * 255.0

    # clip
    # ignores gradient = 1 for now
    x = tf.clip_by_value(x, 0.0, 255.0);
    decoder_output = tf.math.floor(x)

    decoder = tf.keras.Model(inputs=encoder_output, outputs=decoder_output, name="CAE_model")

    # autoencoder.summary()
    # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #                     loss=dict(rate=pass_through_loss, distortion=pass_through_loss),
    #                     metrics=dict(rate=pass_through_loss, distortion=pass_through_loss),
    #                     loss_weights=dict(rate=1., distortion=lmbda),)

    return [encoder, decoder]


class MNISTCompressionTrainer(tf.keras.Model):
  def __init__(self, latent_dims):
    super().__init__()
    [self.encoder, self.decoder] = CreateModel()
    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))

  @property
  def prior(self):
    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    # Ensure inputs are floats in the range (0, 1).
    x = tf.cast(x, self.compute_dtype) / 255.
    x = tf.reshape(x, (-1, 28, 28, 1))

    # Compute latent space representation y, perturb it and model its entropy,
    # then compute the reconstructed pixel-level representation x_hat.
    y = self.encoder(x)
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=1, compression=False)
    y_tilde, rate = entropy_model(y, training=training)
    x_tilde = self.decoder(y_tilde)

    # Average number of bits per MNIST digit.
    rate = tf.reduce_mean(rate)

    # Mean absolute difference across pixels.
    distortion = tf.reduce_mean(abs(x - x_tilde))

    return dict(rate=rate, distortion=distortion)


if __name__ == "__main__":
    latent_dims = 50
    CAE = MNISTCompressionTrainer(latent_dims)
    CAE.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            # Just pass through rate and distortion as losses/metrics.
            loss=dict(rate=pass_through_loss, distortion=pass_through_loss),
            metrics=dict(rate=pass_through_loss, distortion=pass_through_loss),
            loss_weights=dict(rate=1., distortion=lmbda)
            )

    data_dir = os.path.dirname("data/130k")
    batch_size = 1000

    tv_seed = random.randint(0, 1000000)
    e_seed = random.randint(0, 1000000)

    training_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.25,
        subset="training",
        seed=tv_seed,
        image_size=(512, 512),
        batch_size=batch_size)

    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.25,
        subset="validation",
        seed=tv_seed,
        image_size=(512, 512),
        batch_size=batch_size)

    eval_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.25,
        subset="validation",
        seed=e_seed,
        image_size=(512, 512),
        batch_size=batch_size)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # CAE.load_weights(checkpoint_path)

    CAE.fit(training_data, epochs=32, shuffle=True, validation_data=validation_data, callbacks=[cp_callback])
    CAE.evaluate(eval_data, eval_data, callbacks=[cp_callback])
