"""
Code for One Shot detection of an image
Reference: https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
"""

import os
import cv2
import numpy as np
import numpy.random as rng
from keras import Input, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Lambda
from keras.models import Sequential
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
from keras.optimizers import  Adam
import pickle
from sklearn.utils import shuffle


def load_image(path=""):
    """Function to load train/test image into tensors"""
    lang_dict = {}
    cat_dict = {}
    curr_y = 0
    y = []
    X = []
    print(os.path.isdir(path))
    for alphabet in os.listdir(path):
        print(f"Loading Alphabet:{alphabet}")
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)
            for character in os.listdir(letter_path):
                image_path = os.path.join(letter_path, character)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)

            curr_y += 1
            lang_dict[alphabet][1] = curr_y-1

    y = np.vstack(y)
    X = np.stack(X)

    return X, y, lang_dict


class siamese_model_class:
    def __init__(self, path, data_subsets = ['train','val']):
        self.data = {}
        self.categories = {}
        for name in data_subsets:
            file_path = os.path.join(path, name+".pickle")
            with open(file_path, "rb") as f:
                (X,c) = pickle.load(f)
            self.data[name] = X
            self.categories[name] = c

    def get_batch(self, batch_size, s="train"):
        """This method is used to generate random batch of training sets / test sets"""
        X = self.data[s]
        categories = self.categories[s]

        n_classes, n_examples, w, h = X.shape

        categories = rng.choice(n_classes, size = batch_size, replace=False)

        pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
        targets = np.zeros((batch_size, ))

        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = np.random.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = np.random.randint(0, n_examples)

            if i >= batch_size//2:
                category_2 = category
            else:
                category_2 = (category+np.random.randint(1, n_classes)) % n_classes

            pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

        return pairs, targets

    def generate_batch(self, batch_size, is_train=True):
        """This method is used to yield the train/test set"""

        while True:
            pairs, targets = self.get_batch(batch_size, is_train)
            yield  pairs, targets

    def initialize_weights(self, shape, name=None):

        return np.random.normal(loc = 0.0, scale=1e-2, size=shape)

    def initialize_bias(self, shape, name=None):

        return np.random.normal(loc = 0.5, scale=1e-2, size=shape)

    def siamese_model(self, input_shape):
        """Function defining the Simaese model architecture"""
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        model = Sequential()
        model.add(
            Conv2D(64, (10, 10), activation='relu', input_shape = input_shape,
                   kernel_initializer= initializers.random_normal(stddev=1e-2), kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(
            Conv2D(128, (7, 7), activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-2),
                         bias_initializer= initializers.random_normal(stddev=1e-2, mean=0.5),
                         kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(
            Conv2D(128, (4, 4), activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-2),
                   bias_initializer=initializers.random_normal(stddev=1e-2, mean=0.5),
                   kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(
            Conv2D(256, (4, 4), activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-2),
                   bias_initializer=initializers.random_normal(stddev=1e-2, mean=0.5),
                   kernel_regularizer=l2(2e-4)))
        model.add(Flatten())

        model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initializers.random_normal(stddev=1e-2),
                        bias_initializer=initializers.random_normal(stddev=1e-2, mean=0.5)))

        encoder_l = model(left_input)
        encoder_r = model(right_input)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0]-tensors[1]))
        L1_distance = L1_layer([encoder_l,encoder_r])

        prediction = Dense(1, activation='sigmoid',bias_initializer=initializers.random_normal(stddev=1e-2, mean=0.5))(L1_distance)

        siamese_net = Model(inputs = [left_input, right_input], outputs = prediction)

        return siamese_net

    def make_oneshot_task(self, N, s="val", language=None):
        X = self.data[s]
        categories = self.categories[s]

        n_classes, n_examples, w, h = X.shape

        indices = rng.randint(0, n_examples, size=(N,))
        if language is not None:
            low, high = categories[language]
            if N > high-low:
                raise ValueError("Not enough characters")
            categories = rng.choice(range(low, high), size=(N, ), replace=False)
        else:
            categories = rng.choice(range(n_classes), size=(N,), replace=False)

        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1, :, :]]*N).reshape(N, w, h, 1)
        support_set = X[categories, indices, :, :]
        support_set[0,:,:] = X[true_category, ex2]
        support_set = support_set.reshape(N, w, h, 1)
        targets = np.zeros((N, ))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def test_one_shot(self, model, N, k, s="val"):
        """Find the accuracy of the model"""
        print(f"Evaluating One shot model on {k} random {N} way one-shot learning tasks")
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1

        percent_correct = (100 * n_correct)/k
        print(f"Accuracy of One shot model is {percent_correct}")
        return percent_correct

    def train_on_batch(self, model, batch_size):
        model.fit_generator(self.generate_batch(batch_size))

def store_input_data():
    X, y, c = load_image("images_evaluation")

    with open("data/val.pickle", "wb") as f:
        pickle.dump((X, c), f)

    X, y, c = load_image("images_background")

    with open("data/train.pickle", "wb") as f:
        pickle.dump((X,c), f)

def train_model():
    evaluate_every = 10
    loss_every = 20
    batch_size = 32
    n_iter = 20000
    N_way = 20
    n_val = 250
    best = -1
    weights_path = os.path.join("weights","weights.hd5")


    loader = siamese_model_class("data")

    print("Get Model")
    model = loader.siamese_model((105,105,1))
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    print("Start Training process\n------------------\n")
    for i in range(n_iter):
        inputs, targets = loader.get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        print("----------------------")
        if i % evaluate_every == 0:
            print("Validation")
            val_acc = loader.test_one_shot(model, N_way, n_val)
            if val_acc >= best:
                print(f"Current Best:{val_acc:.2f}, Previous Best:{best:.2f}")
                print(f"Saving Weights to:{weights_path}")
                model.save_weights(weights_path)
                best = val_acc

        if i % loss_every == 0:
            print(f"Iteration:{i}, training Loss:{loss:0.2f}")



train_model()





















