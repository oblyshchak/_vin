# NN for classify VIN images


This project involves designing a neural network that can classify small squared black and white images containing a single handwritten character, specifically VIN character boxes.

## Files
- train.py: Python script contating the preprocessed dataset with images, train and save NN model.
- inference.py: Python script contating the getting data from CLI, preprocessed test image and predict class/label image.
- model.h5: Binary ready model.
- requirements.txt: TXT file with the necessary frameworks.
- readme.md:Description of the project and instructions for using the project.
- Dockerfile

## How to run:
- build --> ```docker build . -t vin```
- run --> ```docker run -v ${PWD}/test_images:/mnt/test_images/ -it vin python3 /app/inference.py --input /mnt/test_images/```

## Data 
The EMNIST dataset is used, which has been filtered to include only uppercase letters and numbers. 

## Preprocessing
Data was filtered, because VIN code contain only upper letters and numbers.
The provided images undergo a preprocessing step before being used for training the neural network. The `preprocess` function applies the following transformations to each image:

1. Rotate the image by -90 degrees.
2. Flip the image horizontally.
3. Convert the image to float type and scale the pixel values between 0 and 1.
4. Resize the image to a dimension of 28x28.

The function returns the preprocessed image and its corresponding label.

## Model Architecture
The neural network model is built using the Keras library. The architecture consists of the following layers:

1. Convolutional layer with 32 filters, a kernel size of 3x3, and ReLU activation.
2. Max pooling layer with a pool size of 2x2.
3. Convolutional layer with 64 filters, a kernel size of 3x3, and ReLU activation.
4. Max pooling layer with a pool size of 2x2.
5. Flatten layer to convert the 2D feature maps into a 1D feature vector.
6. Dropout layer with a rate of 0.4 to prevent overfitting.
7. Dense layer with 512 units and ReLU activation.
8. Dense layer with 128 units and ReLU activation.
9. Dense layer with 36 units (corresponding to the number of classes) and softmax activation.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric. The training is performed for 30 epochs.

## Training Results

After training the model, the following accuracies were achieved:

- Train set accuracy: 0.93
- Validation set accuracy: 0.9
- Test set accuracy: 0.91

## Inference Script

To perform inference on a folder of images and predict their corresponding classes, an inference script (`inference.py`) is provided. The script accepts the path to the image folder as a command-line argument. It preprocesses the images using the same preprocessing steps described earlier and outputs the predicted class for each image.


