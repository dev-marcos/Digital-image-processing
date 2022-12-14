# --- Library Imports ---
import numpy as np
import scipy as sp
import pandas as pd

# Ski-Learn
# To divide the data between train and test
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Classfication Report and Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

from skimage.io import imread
from skimage.filters import prewitt_h, prewitt_v

# OpenCV-Python
import cv2

# Visualisation with MatPlotLib
import matplotlib.pyplot as plt

# When using python notebook, un-comment line below
# %matplotlib inline


# --- Important Definitions ---

# Defining tha path for the images
IMAGE_DATA_PATH = "./images/"

# Train and Teste datas are defined in a .csv
train_data = pd.read_csv("./train.csv")
# But only the train.csv is used because the Ski-Learn alredy separates between train and test
test_data = pd.read_csv("./test.csv")

# Category of the images (0,1,2,3,4,5,6,7,8)
categories = [
    "Apple",
    "Avocado",
    "Banana",
    "Grape",
    "Guava",
    "Mango",
    "Orange",
    "Pear",
    "Pineapple",
]

# Input array with flattened images
flat_data_arr = []
flat_data_arr_gray = []
flat_data_arr_mean = []
flat_data_arr_horizontal_edges = []
flat_data_arr_vertical_edges = []

# Get the mean of a image
def mean_image(image):
    feature_matrix = np.zeros((720, 720))

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            feature_matrix[i][j] = (
                int(image[i, j, 0]) + int(image[i, j, 1]) + int(image[i, j, 2])
            ) / 3

    return feature_matrix


# Function to load the training images
def load_image(image_id):
    file_path = image_id + ".png"
    # Read
    image = imread(IMAGE_DATA_PATH + file_path)
    # Resize - "Since SVM receives inputs of the same size, all images need to be resized to a fixed size before inputting them to the SVM"
    image = cv2.resize(image, (720, 720))
    # Flatten the image - Same as image.reshape(-1)
    flat_data_arr.append(image.flatten())
    # Grayscale image flattened
    flat_data_arr_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten())
    # Mean
    flat_data_arr_mean.append(mean_image(image).flatten())
    # Horizontal
    # flat_data_arr_horizontal_edges.append(
    #     prewitt_h(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).flatten()
    # )
    # # Vertical
    # flat_data_arr_vertical_edges.append(
    #     prewitt_v(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).flatten()
    # )

    return image


# *** This has to be called to fill the flat_data_arr ***
# Images from train dataset, for showing purposes
print("Getting images, it takes a while now, because of mean")
train_images = train_data["image_id"].apply(load_image)

# To np.array
flat_data = np.array(flat_data_arr)
flat_data_gray = np.array(flat_data_arr_gray)
flat_data_mean = np.array(flat_data_arr_mean)
flat_data_horizontal = np.array(flat_data_arr_horizontal_edges)
flat_data_vertical = np.array(flat_data_arr_vertical_edges)

# Plotting multiple images using subplots
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 16))
for col in range(3):
    for row in range(2):
        ax[row, col].imshow(train_images.loc[train_images.index[row * 3 + col]])
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])


def showTrainImages():
    print("Loading images...")
    plt.show()


# --- Classifier ---
# Get the expect result from train_data
y = np.array(train_data["class"])

flat_data = np.append(flat_data_gray, flat_data_mean, axis=0)
y = np.append(y, y, axis=0)

# I dont fully know why how this works. I just know that this is the for that the data
# should be input in the classifier
df = pd.DataFrame(flat_data)  # dataframe
df["Target"] = y
X = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data

# # The thing above should do the same thing as this, but dont work.
# X = flat_data.reshape(len(y), -1)
# y = y.reshape(-1)

# Separate the data between test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("Training data and target sizes: \n{}, {}".format(X_train.shape, y_train.shape))
print("Test data and target sizes: \n{}, {}".format(X_test.shape, y_test.shape))


# # SVC classifier, with linear kernel
svclassifier = SVC(kernel="linear", probability=False)
# # GridSearchCV
# param_grid = {
#     "C": [0.1, 1, 10, 100],
#     "gamma": [0.0001, 0.001, 0.1, 1],
#     "kernel": ["rbf", "poly"],
# }
# svc = SVC(probability=False)
# svclassifier = GridSearchCV(svc, param_grid, n_jobs=-1)

print("Started train of SVC model...")
# Train the classifier
svclassifier.fit(X_train, y_train)
print("Finished train.")

y_pred = svclassifier.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))
