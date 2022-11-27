from model import create_model
from model import checkpoint_path

import tensorflow as tf
import glob
import cv2
import numpy
import random
import gc

if __name__ == "__main__":
    CAE = create_model()

    CAE.load_weights(checkpoint_path)

    files = glob.glob("./data/full/*/*.png", recursive=True)

    X_data = []

    random.seed(1024)  # constant seed to ensure same evaluation set between runs.
    used_data = random.sample(range(1, len(files)), 13750)

    for i in range(0, len(used_data)):
        file = files[used_data[i]]
        image = cv2.imread(file)
        X_data.append(image / 255.0)

    del files
    del used_data
    gc.collect()

    X_data = numpy.array(X_data, dtype='float32')

    model_output = CAE(X_data)
    model_output = model_output * 255
    model_output = model_output.numpy()
    model_output = model_output.astype('uint8')

    for i in range(0, len(X_data)):
        cv2.imwrite("validation_input" + str(i) + ".png", X_data[i] * 255.0)

    for i in range(0, len(model_output)):
        cv2.imwrite("validation_output" + str(i) + ".png", model_output[i])

"""
    CAE_input = CAE.input
    encoded_output = CAE.layers[19].output
    print(encoded_output.shape)
    encoder = tf.keras.models.Model(CAE_input, encoded_output)

    encoded_data = encoder(X_data)
    # calculate entropy

    del encoded_data
    gc.collect()
"""
