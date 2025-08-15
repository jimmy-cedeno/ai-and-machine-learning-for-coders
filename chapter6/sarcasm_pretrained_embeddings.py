import json
import tensorflow as tf
import numpy as np
import string
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup


stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
             'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
             'he', 'hed', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how',
             'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
             'lets', 'me', 'more', 'most', 'my', 'myself', 'nor', 'of', 'on', 'once', 'only', 'or', 'other', 'ought',
             'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'shed', 'shell', 'shes', 'should',
             'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
             'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through',
             'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'wed', 'well', 'were', 'weve', 'were',
             'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why',
             'whys', 'with', 'would', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself',
             'yourselves']

table = str.maketrans('', '', string.punctuation)

with open('sarcasm.json', 'r') as file:
  datastore = json.load(file)

sentences = []
labels = []
urls = []
for item in datastore:
  sentence = item['headline'].lower()
  sentence = sentence.replace(',', ' , ')
  sentence = sentence.replace('.', ' . ')
  sentence = sentence.replace('-', ' - ')
  sentence = sentence.replace('/', ' / ')
  soup = BeautifulSoup(sentence)
  sentence = soup.get_text()
  words = sentence.split()
  filtered_sentence = ''
  for word in words:
    word = word.translate(table)
    if word not in stopwords:
      filtered_sentence = filtered_sentence + word + ' '
  sentences.append(filtered_sentence)
  labels.append(item['is_sarcastic'])
  urls.append(item['article_link'])

training_size = 24000
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
  
hub_layer = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1', output_shape=[20], input_shape=[], dtype=tf.string, trainable=False)

model = tf.keras.Sequential([
  hub_layer,
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

history = model.fit(training_sentences, training_labels, epochs=50, validation_data=(testing_sentences, testing_labels), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel('Epochs')
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

test_sentences = ['grany starting to fear spiders in the garden might be real',
'game of thrones season finale showing this sunday night',
'TensorFlow book will be a best seller']

predictions = model.predict(test_sentences)
print(predictions)