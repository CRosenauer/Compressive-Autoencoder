from model import create_model
from model import checkpoint_path

import tensorflow as tf
import glob
import cv2
import numpy
import random
import os
import gc
import math


def Entropy(data):
    prob = dict()

    size = 16*16*96
    inv_size = 1/size

    for i in range(0, 16):
        for j in range(0, 16):
            for k in range(0, 96):
                prob[data[i, j, k]] = 0

    for i in range(0, 16):
        for j in range(0, 16):
            for k in range(0, 96):
                prob[data[i, j, k]] = prob[data[i, j, k]] + inv_size

    entropy = 0.0

    for k, v in dict.items():
        entropy = entropy + v * math.log2(1 / v)

if __name__ == "__main__":
    CAE = create_model()
    CAE.load_weights(checkpoint_path)

    CAE_input = CAE.input
    encoded_output = CAE.layers[20].output
    print(encoded_output.shape)
    encoder = tf.keras.models.Model(CAE_input, encoded_output)

    print("Detecting input files. This may take some time.")
    files = glob.glob("./data/full/*/*.png", recursive=True)

    X_data = []

    eval_set_size = 13750

    random.seed(1024)  # constant seed to ensure same evaluation set between runs.
    used_data = random.sample(range(1, len(files)), eval_set_size)
    
    print("Loading input files. This may take some time.")
    for i in range(0, len(used_data)):
        file = files[used_data[i]]
        image = cv2.imread(file)
        X_data.append(image / 255.0)

    del files
    del used_data
    gc.collect()

    X_data = numpy.array(X_data, dtype='float32')

    print("Beginning generating output files.")

    num_batches = 250
    batch_size = int(eval_set_size / num_batches)

    write_input_file_dir = r"F:\School\ENSC 424\ML-Project\validation\validation_input"
    write_output_file_dir = r"F:\School\ENSC 424\ML-Project\validation\validation_output"

    total_entropy = 0

    for i in range(0, num_batches):
        start = i * batch_size
        end = (i+1) * batch_size
        data = X_data[start:end]

        model_output = CAE(data)
        model_output = model_output * 255
        model_output = model_output.numpy()
        model_output = model_output.astype('uint8')

        os.chdir(write_input_file_dir)
        for j in range(0, batch_size):
            cv2.imwrite("validation_input" + str(i * batch_size + j) + ".png", data[j] * 255.0)

        os.chdir(write_output_file_dir)
        for j in range(0, len(model_output)):
            total_entropy + Entropy(model_output[j])
            cv2.imwrite("validation_output" + str(i * batch_size + j) + ".png", model_output[j])

    average_entropy = total_entropy / eval_set_size
    print(average_entropy)

"""
    encoded_data = encoder(X_data)
    # calculate entropy

    del encoded_data
    gc.collect()
"""
