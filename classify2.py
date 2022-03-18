import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("data.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "c", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=500, train_size = 3500, random_state=42)
xTrainScaled = xTrain/255
xTestScaled = xTest/255

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xTrainS, yTrain)

def getPrediction(image):
    pil = image.open(image)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = pil.convert("l")
    image_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    
    # converting to scalar quantity
    minimum_pixel = np.percentile(image_resized, pixel_filter)

    # using clip to limit values to 0-255
    image_inverted_scaled = np.clip(image_resized - minimum_pixel, 0, 255)
    maximum_pixel = np.max(image_resized)

    # convert into array
    image_inverted_scaled = np.asarray(image_inverted_scaled/maximum_pixel)
    testsample = np.array(image_inverted_scaled).reshape(1, 784)
    prediction = clf.predict(testsample)
    return prediction[0]

