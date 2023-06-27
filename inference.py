import argparse
import numpy as np
import os

from PIL import Image
from tensorflow import keras

# Label index to ASCII code mapping for EMNIST
CLASS_MAPPING = {
    0 :48,
    1 :49,
    2 :50,
    3 :51,
    4 :52,
    5 :53,
    6 :54,
    7 :55,
    8 :56,
    9 :57,
    10 :65,
    11 :66,
    12 :67,
    13 :68,
    14 :69,
    15 :70,
    16 :71,
    17 :72,
    18 :73,
    19 :74,
    20 :75,
    21 :76,
    22 :77,
    23 :78,
    24 :79,
    25 :80,
    26 :81,
    27 :82,
    28 :83,
    29 :84,
    30 :85,
    31 :86,
    32 :87,
    33 :88,
    34 :89,
    35 :90,
}



def preprocess_image(image_path):
    """ function to preprosecc images in folder """
    # convert to gray
    img = Image.open(image_path).convert('L')
    # resize images
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1))
    # tranform to float and rescaling 0-1
    img_array = img_array.astype('float32') / 255.0
    return img_array


def main():
    """ Main entry point """
    # download model
    path_model = 'model.h5'
    model = keras.models.load_model(path_model)

    # create ArgumentPareser object to getting path from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    path_images = args.input
    if path_images is None:
        print("Path is required")
        exit(1)

    # get a list containing the names of the entries in the directory given by path
    images = os.listdir(path_images)

    # loop for forecasting the label
    for image in images:
        image_path = os.path.join(path_images, image)
        img = preprocess_image(image_path)
        predictions = model.predict(img, verbose = 0)
        predicted_class = np.argmax(predictions)
        label = CLASS_MAPPING[predicted_class]
        #0chr(label), /path/
        print(f"{label:03}, {image_path}")


if __name__ == "__main__":
    main()
