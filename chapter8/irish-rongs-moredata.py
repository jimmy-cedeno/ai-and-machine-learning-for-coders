import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

tokenizer = Tokenizer()
max_sequence_len = 6
sentences = []
alltext = []
data = open('irish-lyrics-eof.txt').read()
corpus = data.lower()
alltext.append(corpus)
words = corpus.split(' ')
range_size = len(words) - max_sequence_len
for i in range(0, range_size):
  thissentence = ''
  for word in range(0,max_sequence_len-1):
    word = words[i+word]
    thissentence += word
    thissentence += ' '
  sentences.append(thissentence)

oov_tok = ''
vocab_size = 2700
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, split=' ', char_level=False)
tokenizer.fit_on_texts(alltext)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in sentences:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# pad sequences
# max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(xs.shape)
print(xs[:20])

model = Sequential()
model.add(Embedding(total_words, 16, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)
model.save('bidiirish2.h5')

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

seed_text = 'sweet jeremy saw dublin'
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
print(model.predict(token_list))
# predicted = model.predict_classes(token_list)
predicted = np.argmax(model.predict(token_list), axis=-1)
pred_classes = model.predict(token_list)
print(pred_classes.reshape(-1)[predicted])
print(predicted)

for word, index in tokenizer.word_index.items():
  if index == predicted:
    print(word)
    break

seed_text = 'are you feeling lucky'
next_words = 100

for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  # predicted = model.predict_classes(token_list, verbose=0)
  predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
  output_word = ''
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  seed_text += ' ' + output_word

print(seed_text)