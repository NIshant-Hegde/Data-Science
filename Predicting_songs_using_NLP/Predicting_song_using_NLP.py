import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Reading from the file and making a list of lyrics split by newline

with open("Perfect.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
content = [x.lower() for x in content]

#Hyperparameters

vocab_size = 100000
oov_token = '<OOV>'
embed_dim = 16
maxlen = 15

#Tokenize the lyrics 
tokenizer = Tokenizer(oov_token = oov_token)
sentences = tokenizer.fit_on_texts(content)
total_words = len(tokenizer.word_index) + 1
print(total_words)

#sequences = tokenizer.texts_to_sequences(sentences)
#padded_seq = pad_sequences(sequences, max_length = )
input_seq = []
for line in content:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]    #Bascially a bi-gram taking two tokens at a time
        input_seq.append(n_gram_sequence)
        print("Token List: {}, N gram sequence: {}".format(token_list,n_gram_sequence))

max_len_sequence = max([len(x) for x in input_seq])
padded_input_seq = pad_sequences(input_seq, maxlen = max_len_sequence, padding = 'pre')

train_x, train_y = padded_input_seq[:,:-1], padded_input_seq[:,-1]
train_y = tf.keras.utils.to_categorical(train_y, num_classes = total_words)  #One hot encoding of labels. The total number of classes equals total_words
train_y[0]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM


model = Sequential([
    Embedding(total_words, 64, input_length = max_len_sequence - 1),
    Bidirectional(LSTM(32)),
    Bidirectional(LSTM(20, return_sequences = True)),
    Dense(total_words, activation = 'softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(train_x, train_y, verbose = 1, epochs = 500)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


test_text = "Dikshit found a love for me \n Darling, just dive right in and follow my lead \n Well, Kissed her on the neck and then I took her by the hand \n Oh, I never knew you were the someone waiting for me \n Cause we were just kids when we fell in love \n Not knowing what it was \n But ain't nobody love you like I do \n But darling, just kiss me slow"
next_words = 100

for _ in range(130):
    token_list = tokenizer.texts_to_sequences([test_text])[0]
    token_list = pad_sequences([token_list], maxlen = max_len_sequence - 1, padding = 'pre')
    predicted = model.predict_classes(token_list, verbose = 0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    test_text += " " + output_word

print(test_text)
