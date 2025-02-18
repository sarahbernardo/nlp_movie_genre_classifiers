# Sarah Bernardo
# CS 4120
import numpy as np

class LogisticRegression:
    """
    This class creates an instance of the Logistic Regression classifier.
    """
    def __init__(self, learning_rate: float, num_iterations: int) -> None:
        """
        Initialization function for the classifier.
        Args:
            learning_rate (float): Adjusts alpha to learn through gradient updates.
            num_iterations (int): Adjusts the number of iterations to train the dataset on.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None

    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Given a set of scores, normalize them to sum up to 1.
        Compare this with your implementation from HW 1.

        Args:
            x (np.ndarray): Individual scores corresponding to each class.

        Returns:
            np.ndarray: Probabilities corresponding to each class.
        """
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  

    
    def _init_weights(self, X: np.ndarray, y: list) -> None:
        """Initializes a weight matrix and biases to 0.

        Args:
            X (np.array): The sparse tfidf matrix.
            y (list): List of labels of each document.
        """
        
        num_docs = len(set(y))
        num_tokens = X.shape[1]
        
        # create a 2-D array to store our weights
        self.weights = np.zeros((num_docs, num_tokens))

        # set up mapping for labels to help woth multinomial classification
        # convert labels to vectors with a single 1 at the index corresponding to the label
        self.label_mapping = {label: i for i, label in enumerate(set(y))}

    
    def _get_label_as_vector(self, y: list) -> np.ndarray:
        """
        "One hot" encodes an input list of labels. 
        This function takes a list of string labels and coverts each into
        a vector that has one 1 at the index corresponding to the label.
        The rest of the items in the vector are 0s.

        Args:
            y (list): List of labels of each document.

        Returns:
            list : a list of lists encoding the class for each document.
        """
        encoded = []

        for label in y:
            # only uses labels from training data
            if label not in self.label_mapping:
                raise ValueError(f'Label {label} not found in training data.')
            # one hot encodes all labels in given list using class object's labels
            else:
                one_hot = [0 if i != self.label_mapping[label] else 1 for i in range(len(self.label_mapping))]
                encoded.append(one_hot)
        return np.array(encoded)

    
    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The cross entropy loss to compute distance between the golden labels and the predicted estimates.
        Calculates loss for multiple examples at once.

        Args:
            y_true (np.ndarray): One hot encoded vector indicating the true label. 
            y_pred (np.ndarray): Probability estimates corresponding to each class.

        Returns:
            float: Loss quantifying the distance between the gold labels and the estimates.
        """

        num_classes = y_true.shape[0]
        loss = np.sum(-(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) / num_classes

        return loss
        

    def train(self, X: np.ndarray, y: list, verbose: bool = False) -> None:
        """
        Trains the model for a certain number of iterations. 
        Gradients must be computed and the weights/biases must be updated at the end of each iteration.
        You need not loop through individual documents - use matrix operations to compute losses instead.

        Hint: you'll need to be careful about matrix dimensions.
        To get the transpose of a matrix, you can use the .T attribute.
        e.g., X.T will return the transpose of X.
        
        Args:
            X (np.array): The sparse tfidf matrix as a numpy array.
            y (list): List of labels of each document.
            verbose (bool): If True, print the epoch number and the loss after each 100th iteration.
        """
        # add a 1 to the end of each row in the X matrix to account for the bias term
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # initialize weights
        self._init_weights(X, y)
        
        # encode labels
        encoded_y = self._get_label_as_vector(y)

        # store classes/labels
        self.classes = y

        if verbose:
            print(f"Training for {self.num_iterations} iterations")
            print("class mappings: ", self.label_mapping)

        for i in range(self.num_iterations):

            # compute the predictions for ALL documents
            z = X.dot(self.weights.T)
            preds = self._softmax(z)

            # compute losses and error
            loss = self._cross_entropy_loss(encoded_y, preds)
            err = preds - encoded_y
            
            # print the loss after each 100th iteration if verbose is True
            if verbose and i in range(0, self.num_iterations, 100):
                print(f'At iteration {i}, loss is: {loss}')

            # compute gradients, update weights/biases
            grads = err.T.dot(X)
            self.weights -= self.learning_rate*grads


    def predict(self, X: np.ndarray) ->str:
        """Create a function to return the genre a certain document vector belongs to.

        Args:
           X (np.array): The sparse tfidf vector for a single example to be labeled as a numpy array.

        Returns:
            str: A human readable class fetched from self.label_mapping
        """       
        # calculate the z values for the document
        X = np.vstack((X, [1]))
        z = X.T.dot(self.weights.T)
        
        # reshape array to fit softmax function
        np.reshape(z, (1, -1))

        # feed z value to softmax function for predictions
        prediction = self._softmax(z)

        # map the prediction to a class
        predicted_label = np.argmax(prediction, axis=1)[0]

        # translate the labels back to human readable form
        for genre, label in self.label_mapping.items():
            if label == predicted_label:
                return genre