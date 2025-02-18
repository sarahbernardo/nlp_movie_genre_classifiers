# Sarah Bernardo
# CS 4120, Spring 2025

import numpy as np
from collections import Counter
import nltk
import string

def tokenize(sentence:str)->list:
    """
    This function tokenizes a sentence using nltk.
    Args:
        sentence (str): The sentence to be tokenized.
    Returns:
        list: A list of tokens.
    """
    tokenized_sentence = nltk.word_tokenize(sentence)
    return tokenized_sentence


class NaiveBayesClassifier:
    """ 
    This class creates an instance of the Naive Bayes classifier.
    """
    def __init__(self):
        self.classes = set()
            

    def train(self, X:list, y:list)->None:
        """
        Computes the prior probabilities of each class and the likelihood probabilities of each word in each class,
        and stores them within object variables.
        These will then be used while predicting the genre of a new movie plot.

        Args:
            X (list): A list of movie plots.
            y (list): A list of target genres corresponding to the movie plots.
        """
        self.classes = set(y)
        self.X_count = Counter(X)
        self.y_count = Counter(y)

        # calculate priors
        self.priors = {}
        for genre in self.y_count:
            prob = self.y_count[genre] / sum(self.y_count.values())
            self.priors[genre] = prob

        # create dict with all genres to store tokens for each class 
        self.all_tokens = {genre: [] for genre in self.classes}

        # adds all words in the plot as individual strings to genre token list
        for plot,genre in zip(X,y):
            plot = plot.lower()
            plot = plot.translate(str.maketrans('', '', string.punctuation))
            plot = plot.split()
            self.all_tokens[genre] += plot

        # compile all words in X into one list for defining vocab (set) and vocab size
        full_lst = []
        for genre in self.all_tokens:
            full_lst += self.all_tokens[genre]

        self.vocab = set(full_lst)
        self.vocab_size = len(self.vocab)

        # initializes list for storing each genre's word probabilities
        self.X_probs = {}
        # initializes list for storing each genre's counter
        self.class_count = {}

        # compile list (self.class_counter)  of counters for each class's token frequencies 
        # and list (self.X_probs)  for token probabilities within a class
        for genre in self.all_tokens:
            # class specific counter
            genre_count = {genre: Counter(self.all_tokens[genre])}
            self.class_count[genre] = Counter(self.all_tokens[genre])

            # total number of tokens in a class
            token_cnt = sum(Counter(self.all_tokens[genre]).values())

            # probability = num times a token appears in that genre / num tokens in the genre
            token_probs = {i: ((genre_count[genre][i] + 1) / (token_cnt + self.vocab_size)) for i in genre_count[genre]}
            # self.X_probs.append({genre: token_probs})
            self.X_probs[genre] = token_probs

        
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        Returns:
            int: The size of the vocabulary.
        """
        return self.vocab_size

    
    def get_classes(self) -> set:
        """
        Returns the set of classes.
        Returns:
            set: The set of classes.
        """
        return self.classes

    
    def get_prior(self, genre:str) -> float:
        """
        Returns the prior probability of a class (genre).
        Args:
            genre (str): The genre for which the prior probability is to be returned.
        Returns:
            float: The prior probability of the genre.
        """
        return self.priors[genre]

    
    def get_likelihood(self, word:str, genre:str):
        """
        Computes the probability of a word given a genre.
        That is, this function returns P(word|genre).

        Always use Laplace smoothing while computing the probability.

        Args:
            word (str): The word for which the likelihood is to be returned.
            genre (str): The genre for which the likelihood is to be returned.
        Returns:
            float: The likelihood of the word given the genre.
        """
        genre = genre.title()
        word = word.lower()

        genre_vocab = self.all_tokens[genre]
        
        if word in genre_vocab:
            return self.X_probs[genre][word]
        else:
            return 1 / (len(genre_vocab) + self.vocab_size)

    
    def predict(self, X: str)-> str:
        """
        Takes in a movie plot.
        Loops over the classes (genres) and computes the probability that the movie belongs to each one.  
        Returns the most likely class.

        Args:
            X (str): A sentence representation of a movie plot.

        Returns:
            str: The genre of the movie plot (label).
        """ 

        # tokenize sentence
        plot = tokenize(X)

        # dct for each genre to be {genre: prob}
        pred_probs = {}

        for genre in self.classes:
            p = 1
            for token in plot:
                if token not in self.vocab:
                    continue 
                p *= self.get_likelihood(token, genre)
            prior = self.get_prior(genre)
            p *= prior
        
            pred_probs[genre] = p

        # get genre with highest prob 
        max_genre = max(pred_probs, key=pred_probs.get)

        return max_genre

