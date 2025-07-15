import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
  'Today is a sunny day',
  'Today is a rainy day',
  'Is it a sunny today?',
  'I really enjoyed walking in the snow today'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = pad_sequences(tokenizer.texts_to_sequences(sentences), padding='post', maxlen=6, truncating='post')
print(sequences)


test_data = [
  'Today is a snowy day',
  'Will it be rainy tomorrow?'
]
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_data), padding='post', maxlen=6, truncating='post')
print(word_index)
print(test_sequences)