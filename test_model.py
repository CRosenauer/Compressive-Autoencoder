from model import create_model
from model import checkpoint_path

import glob
import numpy
import cv2
import tensorflow as tf

if __name__ == "__main__":
    CAE = create_model()
    CAE.load_weights(checkpoint_path)

    files = glob.glob("./data/input test/*.png")

    X_data = []

    for file in files:
        image = cv2.imread(file)
        X_data.append(image / 255.0)

    X_data = numpy.array(X_data, dtype='float32')

    model_output = CAE(X_data)

    model_output = model_output * 255
    model_output = model_output.numpy()
    model_output = model_output.astype('uint8')

    for i in range(0, len(model_output)):
        cv2.imwrite("output" + str(i) + ".png", model_output[i])
