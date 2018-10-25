import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import os


lemmatizer = WordNetLemmatizer()


class Train_data:

    hm_lines = 100000

    def __init__(self, save=False, pickle_data=True, pos_path="tensorflow_tut/pos.txt", neg_path="tensorflow_tut/neg.txt",
                 pickle_file_path="tensorflow_tut/sentiment_set.pickle"):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.pickle_file_path = pickle_file_path
        self.save = save
        self.pickle_data = pickle_data

    def create_lexicon(self):
        lexicon = []
        with open(self.pos_path, 'r') as f:
            contents = f.readlines()
            for l in contents[: Train_data.hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)

        with open(self.neg_path, 'r') as f:
            contents = f.readlines()
            for l in contents[: Train_data.hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)

        lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
        w_counts = Counter(lexicon)
        l2 = []
        for w in w_counts:
                # print(w_counts[w])
            if 1000 > w_counts[w] > 20 and len(w) > 3:
                l2.append(w)

        print(len(l2))
        return l2

    def sample_handling(self, sample, lexicon, classification):

        featureset = []

        with open(sample, 'r') as f:
            contents = f.readlines()
            for l in contents[:Train_data.hm_lines]:
                current_words = word_tokenize(l.lower())
                current_words = [lemmatizer.lemmatize(
                    i) for i in current_words]
                features = np.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1

                features = list(features)
                featureset.append([features, classification])

        return featureset

    def create_feature_sets_and_labels(self, test_size=0.1):
        lexicon = self.create_lexicon()
        features = []
        features += self.sample_handling(self.pos_path, lexicon, [1, 0])
        features += self.sample_handling(self.neg_path, lexicon, [0, 1])
        '''
            the reasons behind shuffle the data not only for statistic  and testing reasons
            but also for traning the neural network because if the data were not shuffle,
            it will be hard for neural network to adjust the weight  since the output here eiter 1,0 or 0,1
        '''
        random.shuffle(features)
        features = np.array(features)

        testing_size = int(test_size * len(features))

        self.train_x = list(features[:, 0][: -testing_size])
        self.train_y = list(features[:, 1][: -testing_size])
        self.test_x = list(features[:, 0][-testing_size:])
        self.test_y = list(features[:, 1][-testing_size:])

        return 0

    def load_data(self):
        if self.pickle_data and os.path.isfile(self.pickle_file_path):
            with open(self.pickle_file_path, 'rb') as f:
                self.train_x, self.train_y, self.test_x, self.test_y = pickle.load(
                    f)
        else:
            self.create_feature_sets_and_labels()
            if self.save:
                self.save_to_pickle()

        return self.train_x, self.train_y, self.test_x, self.test_y

    def save_to_pickle(self):
        # if you want to pickle this data:
        with open(self.pickle_file_path, 'wb') as f:
            pickle.dump([self.train_x, self.train_y,
                         self.test_x, self.test_y], f)

    def next_batch(self, batch_size):
        start = self.current_batch
        end = self.current_batch + batch_size
        batch_x = np.array(self.train_x[start: end])
        batch_y = np.array(self.train_y[start: end])
        self.current_batch += batch_size
        return batch_x,  batch_y

    def num_examples(self):
        self.current_batch = 0
        return len(self.train_x)


csf = Train_data(pickle_data=False)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = Train_data().load_data()
    print(len(train_x))
    print(len(train_y))
    print(len(test_x))
    print(len(test_y))
