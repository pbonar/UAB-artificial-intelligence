__authors__ = ["1000000", "1000000"]
__group__ = 82

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """
        Ensures that X is a float-type matrix with dimensions N × D.
        If the input is an image of dimensions F × C × 3, it reshapes it to (N × 3).
        """
        X = np.array(X, dtype=np.float32)  # (a) Ensure float type

        if len(X.shape) > 2 and X.shape[-1] == 3:
            X = X.reshape(-1, 3)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initializes centroids based on the selected initialization method.
        The available methods are:
        - 'first': Uses the first K distinct points from X.
        - 'random': Selects K unique random points from X.
        """
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))  # Initialize old centroids

        if self.options['km_init'].lower() == 'first':
            # Select the first K distinct points
            seen = set()
            centroids = []
            for point in self.X:
                tuple_point = tuple(point)  # Convert to tuple for uniqueness check
                if tuple_point not in seen:
                    seen.add(tuple_point)
                    centroids.append(point)
                if len(centroids) == self.K:
                    break
            
            if len(centroids) < self.K:
                raise ValueError("Not enough unique points to initialize K distinct centroids.")
            
            self.centroids = np.array(centroids, dtype=np.float32)

        elif self.options['km_init'].lower() == 'random':
            # Select K unique random points as centroids
            indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[indices].astype(np.float32)

        else:
            raise ValueError("Invalid initialization method. Choose 'first' or 'random'.")

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Updates the centroids by computing the mean of all points assigned to each centroid.
        """
        self.old_centroids = self.centroids.copy()  # Store the old centroids

        new_centroids = np.zeros_like(self.centroids, dtype=np.float64)  # Use float64 for precision
        for k in range(self.K):
            points = self.X[self.labels == k]  # Select all points belonging to cluster k
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0, dtype=np.float64)  # Compute mean with high precision

        self.centroids = new_centroids.astype(np.float64)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
