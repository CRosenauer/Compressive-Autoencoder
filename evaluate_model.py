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


def Entropy(data, x_size, y_size, z_size):
    prob = dict()

    size = x_size*y_size*z_size
    inv_size = 1/size

    for i in range(0, x_size):
        for j in range(0, y_size):
            for k in range(0, z_size):
                key = data[i, j, k]
                prob[key] = 0.0

    for i in range(0, x_size):
        for j in range(0, y_size):
            for k in range(0, z_size):
                key = data[i, j, k]
                prob[key] = prob[key] + inv_size

    entropy = 0.0

    for k, v in prob.items():
        entropy = entropy + v * math.log2(1 / v)

    return entropy

if __name__ == "__main__":
    CAE = create_model()
    CAE.load_weights(checkpoint_path)

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
    write_output_file_dir = r"F:\School\ENSC 424\ML-Project\validation\q_r128"

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
            cv2.imwrite("validation_output" + str(i * batch_size + j) + ".png", model_output[j])

    CAE_input = CAE.input
    encoded_output = CAE.layers[20].output
    encoder = tf.keras.models.Model(CAE_input, encoded_output)

    input_entropy = 0.0
    total_entropy = 0.0

    print("Calculating average entropy of encoded images. This may take some time.")
    for i in range(0, num_batches):
        start = i * batch_size
        end = (i+1) * batch_size
        data = X_data[start:end]

        model_output = encoder(data)
        model_output = model_output * 255
        model_output = model_output.numpy()
        model_output = model_output.astype('uint8')

        for j in range(0, len(model_output)):
            total_entropy = total_entropy + Entropy(model_output[j], 16, 16, 96)


        for j in range(0, len(data)):
            input_entropy = input_entropy + Entropy(data[j], 128, 128, 3)

    average_input_entropy = total_entropy / eval_set_size
    average_entropy = input_entropy / eval_set_size

    print("Performing model evaluation")
    CAE.compile(loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])
    CAE.evaluate(X_data, X_data)

    print("Average Input Entropy: " + str(average_input_entropy))
    print("Average Output Entropy: " + str(average_entropy))
    print("Average Bits per Pixel: " + str(1.5 * average_entropy))

    input_storage_size = 128*128*3*average_input_entropy
    print("Average Input Storage Size: " + str(input_storage_size))
    output_storage_size = 16*16*96*average_entropy
    print("Average Encoded Storage Size: " + str(output_storage_size))

    print("Average Compression: " + str(100 * output_storage_size / input_storage_size) + "%")
