"""
This is Simple Bi-LSTM model for Next Word Prediction.

Next Word Prediction (also called Language Modeling) is the task of predicting what word comes next. It is one of the fundamental tasks of NLP.

Language modeling is one of the benchmark tasks of NLP. In its simplest form, it consists of predicting the most probable word following a series of words based on them.

N-Gram language models

Intuitively, in general, more common words like “cat ”or “dog ”should tend to have higher probabilities then more uncommon ones such as aardvark or kingfisher. Thus, a good starting off point could be the frequency of words in the corpus. A system for this purpose that takes into account only the number of appearances of a word normalized by the number of words in the corpus is called a uni-gram language model. Similarly, bi-gram language models consider the frequency of couples of word, for example, if in our English corpus the couple [united, states] appears more often than [united, the] a bi-gram language model would assign a higher probability to “states ” rather then to “the” to follow “united ”despite the much higher frequency of the latter. Higher-gram language models also exist, but as the dimensions of the sequences of words increase, their frequency in the corpus decreases exponentially. These models thus have a sparsity problem and struggle with infrequent word sequences.

LSTM

One of the architecture proposed to solve the vanishing gradient problem is called Long Short-Term Memory, and it works by having both and hidden state and a memory cell and three gates the form read (in the hidden state), write and delete of the cell. They are called the output, input, and forget gate, respectively. This architecture implements a dedicated way to maintain long-term dependencies, which is never forgetting the memory cell.
"""

# Import necessary libraries and packages

import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

class Model_Training:

    __input_sequences = []
    __tokenizer = Tokenizer()
    __max_sequence_len = 0
    __total_words = 0

    def __init__(self, corpus_path:str, model_saving_path:str):
        self.corpus = corpus_path
        self.model_path = model_saving_path

    def __len__(self):
        file = open(self.corpus, "r")
        data = file.read()
        words = data.split()
        return len(words)
    
    def train_tokenize(self):

        file = open(self.corpus,'r')
        sentences = file.readlines()
        self.__tokenizer = Tokenizer(oov_token='<oov>') 
        self.__tokenizer.fit_on_texts(sentences)
        self.__total_words = len(self.__tokenizer.word_index) + 1

        for line in tqdm(sentences):
            token_list = self.__tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            self.__input_sequences.append(n_gram_sequence)

        print("Total number of words: ", self.__total_words)


    def trainmodel(self):
        self.__data_prepartion()

        #check for GPUs
        gpus = tf.config.list_logical_devices('GPU')

        self.__createModel()

        if gpus:
            print("Running Using GPUs")
            self.__model = self.__trainwithGPU(gpus)
        else:
            print("Running Using CPU")
            self.__model = self.__trainwithCPU()



    def __data_prepartion(self):
        # pad sequences
        self.__max_sequence_len = max([len(x) for x in self.__input_sequences])
        self.__input_sequences = np.array(pad_sequences(self.__input_sequences, maxlen=self.__max_sequence_len, padding='pre'))

        # self.__train_dataset = tf.data.Dataset.from_tensor_slices((self.__input_sequences[:,:-1],self.__input_sequences[:,-1]))
        # self.__train_dataset = self.__train_dataset.shuffle(buffer_size=1024).
        # batch(1024)
        target_labels = self.__input_sequences[:, -1]
        self.__target_labels = self.__input_sequences[:, 1:]


    def __createModel(self):

        # model_name = str(input("Enter Name of model:"))
        model_name = "testingmodel"
        from tensorflow.keras.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(filepath='{}/{}.h5'.format(self.model_path,model_name),  # Path to save the model
                                            monitor='accuracy',   # Metric to monitor for improvement
                                            save_best_only=True,      # Save only the best model
                                            mode='max',               # Mode of the monitored metric
                                            verbose=1)

        model = Sequential()
        model.add(Embedding(self.__total_words, 100, input_length=self.__max_sequence_len-1))
        
        backward_layer = LSTM(150, activation='relu', return_sequences=True, go_backwards=True)

        model.add(Bidirectional(LSTM(150,return_sequences=True),backward_layer=backward_layer))

        model.add(Dense(self.__total_words, activation='softmax'))

        adam = Adam(lr=0.01)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model, checkpoint_callback
    
    def __trainwithGPU(self,gpus):

        model, checkpoint_callback = self.__createModel()

        c = []
        for gpu in gpus:
            self.__history = model.fit(self.__train_dataset, batch_size = 512, epochs=150, verbose=1, callbacks=[checkpoint_callback])

        with tf.device('/CPU:0'):
            print(model.summary())

        print("Model is saved in folder {} with name {}")
        
        return model
    
    def __trainwithCPU(self):
        
        model, checkpoint_callback = self.__createModel()
        
        self.__history = model.fit(self.__input_sequences[:,:-1], self.__target_labels, batch_size = 512, epochs=300, verbose=1, callbacks=[checkpoint_callback])

        print(model.summary())

        return model

    def getTrainingData(self):
        return self.__train_dataset
    
    def getTokenizer(self):
        return self.__tokenizer
    
    def getMaxSequenceLength(self):
        return self.__max_sequence_len
    
if __name__ == "__main__":
    mt = Model_Training("../dataset/smallCorpus.txt","../models/")
    print("Total number of words in the corpus is :",len(mt))

    mt.train_tokenize()

    mt.trainmodel()