#!/usr/bin/env python
# coding: utf-8

print("import native packages...", end="", flush=True)
from time import time
import os
import re
import pickle

import random
random.seed(0)
print("done")

print("import numpy...", end="", flush=True)
import numpy as np
np.random.seed(0)
print("done")

print("import pandas...", end="", flush=True)
import pandas as pd
print("done")


print("import sklearn...", end="", flush=True)
# from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import scale
# from nltk.tokenize import RegexpTokenizer
# from sklearn.metrics import confusion_matrix
print("done")

print("import tensorflow...", end="", flush=True)
import tensorflow as tf
print("done")

print("import keras...", end="", flush=True)
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding
from keras.layers.merge import Concatenate
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import concatenate
print("done")

TOTAL_EXAMPLES = 50000 # 50000 or more
SEQ_LEN = 200 # consecutive words per example.
EMBEDDING_DIM = 300 # how big is each word vector
BATCH_SIZE = 128
NUM_FILTERS = 128


W2V_Pickle = "/users/d/m/dmatthe1/OtherProjects/DBs/w2v.p"
DATA_FILE = "/users/d/m/dmatthe1/OtherProjects/DBs/CNN_Wshah_Data"


try:

    X_train_cnn = np.load(DATA_FILE+"X_train_cnn.npx.npy")
    y_train = np.load(DATA_FILE+"y_train.npx.npy")
    X_test_cnn = np.load(DATA_FILE+"X_test_cnn.npx.npy")
    y_test = np.load(DATA_FILE+"y_test.npx.npy")
    print("loaded model from file")
except Exception as e:
    raise (e)
    print("loading w2v")
    print("import gensim...", end="", flush=True)
    import gensim
    from gensim.models.phrases import Phrases, Phraser
    from gensim.models import Word2Vec

    print("done")

    try:
        w2v_model = pickle.load(open(W2V_Pickle, "rb"))
        print("loaded from pickle")
    except:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)
        print("dumping with protocol:", pickle.HIGHEST_PROTOCOL)
        pickle.dump(w2v_model, open(W2V_Pickle, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("loaded from model file")

    w2v_model_tmp = {}

    def get_word_vec(word):

        if word in w2v_model:
            return w2v_model[word]
        elif word in w2v_model_tmp:
            return w2v_model_tmp[word]
        else:
            w2v_model_tmp[word] = np.random.rand(EMBEDDING_DIM)
            return w2v_model_tmp[word]

    print("Done loading w2v")


    banned_txt = "/users/d/m/dmatthe1/OtherProjects/CSV/RC_2016-10_banned.txt"
    notbanned_txt = "/users/d/m/dmatthe1/OtherProjects/CSV/RC_2016-10_notbanned.txt"

    bannedF = open(banned_txt,"r")
    bannedCnt = TOTAL_EXAMPLES

    notbannedF = open(notbanned_txt, "r")
    notbannedCnt = TOTAL_EXAMPLES

    print("%d banned examples and %d notbanned examples"%(bannedCnt, notbannedCnt))

    bannedPosts = [(1, [bannedF.readline().strip() for _ in range(SEQ_LEN)]) for _ in range(bannedCnt)]
    banned_train, banned_test = train_test_split(bannedPosts, test_size=0.1)

    notbannedPosts  = [(0, [notbannedF.readline().strip() for _ in range(SEQ_LEN)]) for _ in range(notbannedCnt)]
    notbanned_train, notbanned_test = train_test_split(notbannedPosts, test_size=0.1)

    banned_train = banned_train

    print("%d banned training examples and %d notbanned training examples"%(len(banned_train), len(notbanned_train)))

    postsListTrain = banned_train
    postsListTrain += notbanned_train

    postsListTest = banned_test
    postsListTest += notbanned_test

    postsTrain= pd.DataFrame(postsListTrain, columns=["banned", "tokens"])
    postsTest = pd.DataFrame(postsListTest, columns=["banned", "tokens"])

    print(postsTrain.head())
    print(postsTest.head())


    y_train = postsTrain["banned"].values
    y_test = postsTest["banned"].values
    X_train = postsTrain["tokens"].values
    X_test = postsTest["tokens"].values

    X_train_cnn = np.zeros((len(X_train), SEQ_LEN, EMBEDDING_DIM))

    X_test_cnn = np.zeros((len(X_test), SEQ_LEN, EMBEDDING_DIM))

    for i in range(X_train_cnn.shape[0]):
        if i %1000==0: print("train", i, len(w2v_model_tmp))
        for j in range(SEQ_LEN):
            word_vec = get_word_vec( X_train[i][j])
            X_train_cnn[i, j, :] = word_vec

    for i in range(X_test_cnn.shape[0]):
        if i %1000==0: print("test", i, len(w2v_model_tmp))
        for j in range(SEQ_LEN):
            word_vec = get_word_vec( X_test[i][j])
            X_test_cnn[i, j, :] = word_vec

    print("X_train_cnn shape:", X_train_cnn.shape )
    print("X_test_cnn shape:", X_test_cnn.shape )
    np.save(DATA_FILE+"X_train_cnn.npx", X_train_cnn)
    np.save(DATA_FILE+"y_train.npx", y_train)
    np.save(DATA_FILE+"X_test_cnn.npx", X_test_cnn)
    np.save(DATA_FILE+"y_test.npx", y_test)
    print("cached model to file")

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation="relu", input_shape=(SEQ_LEN, EMBEDDING_DIM)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate = 0.25))

model.add(Conv1D(filters=128, kernel_size=7, activation="relu", input_shape=(SEQ_LEN, EMBEDDING_DIM)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate = 0.25))

model.add(Conv1D(filters=128, kernel_size=15, activation="relu", input_shape=(SEQ_LEN, EMBEDDING_DIM)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate = 0.25))

model.add(Conv1D(filters=128, kernel_size=15, activation="relu", input_shape=(SEQ_LEN, EMBEDDING_DIM)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])

model.summary()

history = model.fit(X_train_cnn, y_train, epochs=3, batch_size=128,
                    validation_data=(X_test_cnn, y_test))

loss, accuracy = model.evaluate(X_train_cnn, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

exit(1)
classes = [0,1]

print("test data")
y_pred=np.array([1 if prd > 0.5 else 0 for prd in model.predict(X_test_cnn)])
print(y_pred)


sess = tf.compat.v1.Session()

con_mat = sess.run(tf.math.confusion_matrix(labels=y_test, predictions=y_pred))

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index=classes,
                          columns=classes)
print(con_mat)
print(con_mat_df)
print("row: what should have been predicted")
print("column: what was predicted")

print("train data")
y_pred=np.array([1 if prd > 0.5 else 0 for prd in model.predict(X_train_cnn)])
print(y_pred)

sess = tf.compat.v1.Session()

con_mat = sess.run(tf.math.confusion_matrix(labels=y_train, predictions=y_pred))

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index=classes,
                          columns=classes)
print(con_mat)
print(con_mat_df)
print("row: what should have been predicted")
print("column: what was predicted")
