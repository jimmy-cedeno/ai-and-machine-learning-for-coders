import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

from bs4 import BeautifulSoup
import string

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

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
  sentence = str(item['text'].decode('UTF-8').lower())
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
  imdb_sentences.append(filtered_sentence)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

print(tokenizer.word_index)

sentences = [
  'Today is a sunny day',
  'Today is a rainy day',
  'Is it sunny today?'
]

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)


reverse_word_index = dict(
  [(value, key) for (key, value) in tokenizer.word_index.items()]
)

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])
print(decoded_review)


# builder = tfds.builder("imdb_reviews")
# print(builder.builder_configs)


# (train_data, test_data), info = tfds.load(
#     # Use the version pre-encoded with an ~8k vocabulary.
#     'imdb_reviews/plain_text', 
#     # Return the train/test datasets as a tuple.
#     split = (tfds.Split.TRAIN, tfds.Split.TEST),
#     # Return (example, label) pairs from the dataset (instead of a dictionary).
#     as_supervised=True,
#     # Also return the `info` structure. 
#     with_info=True)
# encoder = info.features['text'].encoder

# print('Vocabulary size: {}'.format(tokenizer.vocab_size))

# print(encoder.decode([1, 2, 3]))

# sample_string = 'Today is a sunny day'

# encoded_string = encoder.encode(sample_string)
# print ('Encoded string is {}'.format(encoded_string))

# original_string = encoder.decode(encoded_string)
# print ('The original string: "{}"'.format(original_string))

# assert original_string == sample_string

# print(encoder.subwords[6426])