#!/usr/bin/env python
# coding: utf-8

# # Based on post from:
# https://towardsdatascience.com/natural-language-processing-classification-using-deep-learning-and-word2vec-50cbadd3bd6a

import random
random.seed(0)
import pandas as pd
import numpy as np
np.random.seed(0)

import tensorflow as tf

import os
import re
from gensim.models.phrases import Phrases, Phraser
from time import time
import multiprocessing
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import scale
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding
from keras.layers.merge import Concatenate
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix
import gensim
import pickle

print("Done importing")
# In[2]:

W2V_Pickle = "/users/d/m/dmatthe1/OtherProjects/DBs/w2v.p"

CSV_NAME = "/users/d/m/dmatthe1/OtherProjects/DBs/2016-10-Data.csv"

posts = pd.read_csv(CSV_NAME)
posts["tokens"] = posts["document"].apply(lambda x: x.split(" "))
print(posts.head())



#First defining the X (input), and the y (output)
y = posts['banned'].values
X = np.array(posts["tokens"])

#And here is the train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


print("loading w2v")
try:
    w2v_model = pickle.load(open(W2V_Pickle, "rb"))
    print("loaded from pickle")
except:
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)
    pickle.dump(w2v_model, open(W2V_Pickle, "wb"))
    print("loaded from model file")

print("Done loading w2v")


all_words = [word for tokens in X for word in tokens]
all_sentence_lengths = [len(tokens) for tokens in X]
ALL_VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(ALL_VOCAB)))
print("Max sentence length is %s" % max(all_sentence_lengths))


####################### CHANGE THE PARAMETERS HERE #####################################
EMBEDDING_DIM = 300 # how big is each word vector
MAX_VOCAB_SIZE = len(ALL_VOCAB) # old: 18399# how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH = max(all_sentence_lengths) # old: 53 # max number of words in a comment to use


# In[30]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, char_level=False)

tokenizer.fit_on_texts(posts["tokens"].tolist())
training_sequences = tokenizer.texts_to_sequences(X_train.tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = w2v_model[word] if word in w2v_model else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)


######################## TRAIN AND TEST SET #################################
train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_sequences = tokenizer.texts_to_sequences(X_test.tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


print(train_cnn_data[0].shape)


from keras.layers import concatenate
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, trainable=False, extra_conv=True):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate([convs[0],convs[1],convs[2]],axis=1)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv==True:
        x = Dropout(0.5)(l_merge)  
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) 
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(1,activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])
    model.summary()
    return model


model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, False)

history = model.fit(train_cnn_data, y_train, epochs=3, batch_size=64,
                   validation_data=(test_cnn_data, y_test))

loss, accuracy = model.evaluate(train_cnn_data, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_cnn_data, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

classes = [0,1]

y_pred=np.array([1 if prd > 0.5 else 0 for prd in model.predict(test_cnn_data)])
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
