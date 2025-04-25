import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
from itertools import combinations
import re
from sklearn.metrics import precision_score, make_scorer, accuracy_score

def NonLinearPenalty(prob, confidence, sharpness=10):
    return 2 / (1 + math.exp(-sharpness * (prob - confidence)))

def WordScore(scores):
    product = 1.0
    for s in scores:
        product *= max(s, 1e-6)
    return product ** (1.0 / len(scores)) if scores else 0

def strings_to_position_tokens(strings):

    vocab = sorted(set("".join(strings)))
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)

    result = []
    for s in strings:
        one_hot_string = []
        for char in s:
            one_hot = [0] * vocab_size
            one_hot[vocab_dict[char]] = 1
            one_hot_string.append(one_hot)
        result.append(one_hot_string)

    return result, vocab

def strings_to_char_positions(strings, vocab):
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}
    return [
        [vocab_dict[char] for char in s if char in vocab_dict]
        for s in strings
    ]

class SpeechClassifier_FR(BaseEstimator, ClassifierMixin):
    def __init__(self, p_threshold=0.5, w_threshold=0.5, confidence=0.5, sharpness=1.0):
        self.p_threshold = p_threshold
        self.w_threshold = w_threshold
        self.confidence = confidence
        self.sharpness = sharpness

    def fit(self, X, y):
        self.words_ = X.iloc[:, 0]
        self.targets_ = X.iloc[:, 1]
        self.deletions_ = X.iloc[:, 2]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        words_list = X.iloc[:, 0]
        targets_list = X.iloc[:, 1]
        deletions_list = X.iloc[:, 2] if X.shape[1] > 2 else [None] * len(X)

        predictions = []
        for words, target, deletion in zip(words_list, targets_list, deletions_list):
            result = self.classifier(target, words, deletion)
            predictions.append(int(result))
        return np.array(predictions)

    def classifier(self, target, words, deletion=None):
        S_t = len(target)
        for w in words:
            S_w = len(w)
            if S_w < S_t:
                continue
            for i in range(S_w - S_t + 1):
                Scores = []
                for pw, pt in zip(w[i:i + S_t], target):
                    if deletion is not None and i!=0:
                        if np.argmax(w[i-1]) == deletion:
                            break
                    s = 0
                    prob = pw[pt]
                    if prob > self.confidence:
                        s = 1
                    else:
                        s = NonLinearPenalty(prob, sharpness=self.sharpness, confidence=self.confidence)

                    Scores.append(s)

                if WordScore(Scores) > self.w_threshold:
                    for score in Scores:
                        if score < self.p_threshold:
                            break
                        return True
        return False
    

class SpeechClassifier_IT(BaseEstimator, ClassifierMixin):
    def __init__(self, p_threshold=0.5, w_threshold=0.5, confidence=0.5, sharpness=1.0):
        self.p_threshold = p_threshold
        self.w_threshold = w_threshold
        self.confidence = confidence
        self.sharpness = sharpness

    def fit(self, X, y):
        self.words_ = X.iloc[:, 0]
        self.targets_ = X.iloc[:, 1]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        words_list = X.iloc[:, 0]
        targets_list = X.iloc[:, 1]

        predictions = []
        for words, target in zip(words_list, targets_list):
            result = self.classifier(target, words)
            predictions.append(result)
        return np.array(predictions)


    def label(self, target, word):
        S_t = len(target)
        S_w = len(word)
        if S_w < S_t:
            return False
        for i in range(S_w - S_t + 1):
            Scores = []
            for pw, pt in zip(word[i:i + S_t], target):
                s = 0
                prob = pw[pt]
                if prob > self.confidence:
                    s = 1
                else:
                    s = NonLinearPenalty(prob, sharpness=self.sharpness, confidence=self.confidence)

                Scores.append(s)

            if WordScore(Scores) > self.w_threshold:
                for score in Scores:
                    if score < self.p_threshold:
                        break
                    return True
        return False
    
    def classifier(self,target,words):
        labels = []
        for i, word in enumerate(words):
            label = self.label(target[i], word)
            # For swaps, one element is classified as True and the other as False (Not tested)
            '''if label == False and i != len(words) - 1:
                label = self.label(target[i], words[i+1])'''
            labels.append(label)
        return labels

def create_data(file_path, API_target_path='Phoneme_Deleletion_ground_truth_FR.csv', vocab_path='vocab.json', coder=1, deletion_path=None):

    output = pd.read_csv(file_path)
    output['probabilities'] = output['probabilities'].apply(eval)
    output['timestamps'] = output['timestamps'].apply(eval)
    vocab = pd.read_json(vocab, typ='series').keys().tolist()

    ground_truth_fr = pd.read_csv(API_target_path)[['file_name', 'API_target',f'accuracy_coder{coder}']]
    output = pd.merge(output, ground_truth_fr, left_on='file_name', right_on='file_name')
    output = output[['probabilities','API_target',f'accuracy_coder{coder}']]
    output['API_target'] = output['API_target'].apply(lambda x: re.sub(r'[\[\]]', '', x))

    target = strings_to_char_positions(output['API_target'].astype(str).tolist(), vocab)
    target_series = pd.Series(target, name='target_tokens').reset_index(drop=True)
    if deletion_path is not None:
        X = pd.concat([output['probabilities'].apply(lambda x: [x]).reset_index(drop=True), target_series, pd.read_csv(deletion_path)], axis=1)
    else:
        X = pd.concat([output['probabilities'].apply(lambda x: [x]).reset_index(drop=True), target_series, pd.Series([None] * len(output), name='third_column')], axis=1)
    y = output['accuracy_coder1'].reset_index(drop=True)

    return X, y


def model_train_FR(X, y, param_grid=None):
    if param_grid is None:
        # Default parameter grid
        param_grid = {
            'p_threshold': np.arange(0.1, 1.0, 0.1),
            'w_threshold': np.arange(0.1, 1.0, 0.1),
            'confidence': np.arange(0.1, 1.0, 0.1),
            'sharpness': np.arange(1, 10, 1),
        }

    grid = GridSearchCV(SpeechClassifier_FR(), param_grid, scoring='accuracy',cv=2)
    grid.fit(X, y)

    return grid

def get_predict(file_path, API_target_path='Phoneme_Deleletion_ground_truth_FR.csv', vocab_path='vocab.json', coder=1, deletion_path=None):

    X, y = create_data(file_path, API_target_path, vocab_path, coder, deletion_path)
    model = model_train_FR(X, y)
    predictions = model.predict(X)

    return predictions

