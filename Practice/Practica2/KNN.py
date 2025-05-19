__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
import numpy as np
import math
import operator
import pickle
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        # THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        Initializes the training data:
        - Converts to float32
        - Reshapes images to vectors of size 4800 (80x60)

        Args:
            train_data: PxMxNx3 matrix (P color images of size MxN)

        Result:
            self.train_data: Px4800 matrix (P images, each as a flat vector)
        """
        train_data = np.array(
            train_data, dtype=np.float32)  # Ensure float type

        if train_data.ndim == 4 and train_data.shape[-1] == 3:
            # Convert RGB to grayscale by averaging over channels
            train_data = np.mean(train_data, axis=-1)  # Now shape is P x M x N

        P, M, N = train_data.shape  # Get dimensions
        self.train_data = train_data.reshape(
            P, M * N)  # Flatten each image to 1D

    def get_k_neighbours(self, test_data, k):
        """
        Given a test_data matrix, calculates the k nearest neighbours of each point.

        Args:
            test_data (array): Test data of shape NxMxNx3 (N images)
            k (int): Number of neighbors to retrieve

        Result:
            self.neighbors: NxK matrix. Each row contains the labels of the k nearest training samples.
        """
        # Step 1: Ensure float32 and convert RGB to grayscale (like in _init_train)
        test_data = np.array(test_data, dtype=np.float32)

        if test_data.ndim == 4 and test_data.shape[-1] == 3:
            test_data = np.mean(test_data, axis=-1)  # RGB to grayscale

        # Step 2: Flatten each test image to 1D vector
        N, M, P = test_data.shape  # N test images of size MxP
        test_data_flat = test_data.reshape(N, M * P)

        # Step 3: Compute distances between each test image and all training images
        distances = cdist(test_data_flat, self.train_data,
                          metric='euclidean')  # Shape: N x num_train_samples

        # Step 4: Get the indices of the k smallest distances for each test image
        nearest_indices = np.argsort(distances, axis=1)[:, :k]  # Shape: N x K

        # Step 5: Store the corresponding labels
        self.neighbors = self.labels[nearest_indices]  # Shape: N x K

    def get_class(self):
        """
        Get the predicted class for each test sample by majority voting (without using scipy.stats.mode).

        Returns:
            predicted_classes (np.ndarray): Array of shape (N,) with the predicted class per test sample.
        """
        predicted_classes = []

        for row in self.neighbors:
            # Count occurrences of each class label in the row
            label_counts = {}
            for label in row:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            # Get label with highest count (i.e. majority vote)
            majority_label = max(label_counts.items(),
                                 key=operator.itemgetter(1))[0]
            predicted_classes.append(majority_label)

        return np.array(predicted_classes)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()


def load():
    with open('./test/test_cases_knn.pkl', 'rb') as f:
        test_cases = pickle.load(f)
    return test_cases

def test_init_train():
    with open('./test/test_cases_knn.pkl', 'rb') as f:
        test_cases = pickle.load(f)
    for ix, (train_imgs, train_labels) in enumerate(test_cases['input']):
        knn = KNN(train_imgs, train_labels)
        # print(len(knn.train_data))
        # print(knn.train_data.shape)
        # print(knn.train_data.size)
        knn.get_k_neighbours(test_cases['test_input'][ix][0], 2)
        print('test')

# test_init_train()

with open('./test/test_cases_knn.pkl', 'rb') as f:
    test_cases = pickle.load(f)

# Load dataset in grayscale
data = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

for ix, (train_imgs, train_labels) in enumerate(test_cases['input']):
    knn = KNN(train_imgs, train_labels)
    knn.get_k_neighbours(
    test_cases['test_input'][ix][0], test_cases['rnd_K'][ix])
    preds = knn.get_class()

    # test_img = read_one_img('./imatge2.png',80,60,False)           # the test image to classify
    # predicted_label = knn.predict(preds, 5)

    # print(predicted_label)
    # predicted_label = knn.predict(np.expand_dims(test_img, axis=0), 5)
    # print(predicted_label)


# # Assume the dataset has the following structure:
# train_imgs = data['train_imgs']       # all training images (in grayscale)
# train_labels = data['train_labels']   # labels associated with the training images
# test_img = read_one_img('./Imatge1.png',80,60,False)           # the test image to classify

# # Initialize the classifier using all training images and labels
# knn = KNN(train_imgs, train_labels)

# # Predict the label for the test image using K=5.
# # Note: The predict method expects a batch of test images. If test_img is a single image, add a batch dimension.
# predicted_label = knn.predict(np.expand_dims(test_img, axis=0), 5)
# print(predicted_label)
