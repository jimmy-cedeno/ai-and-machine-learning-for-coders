import tensorflow as tf
import tensorflow_datasets as tfds

data, info = tfds.load('mnist', with_info=True)
filename = '/Users/jimmycedeno/tensorflow_datasets/mnist/3.0.1/mnist-test.tfrecord-00000-of-00001'

raw_dataset = tf.data.TFRecordDataset(filename)
for raw_record in raw_dataset.take(1):
  print(repr(raw_record))

# Create a description of the features
feature_description = {
  'image': tf.io.FixedLenFeature([], dtype=tf.string),
  'label': tf.io.FixedLenFeature([], dtype=tf.int64)
}

def _parse_function(example_proto):
  # Parse the input `tf.example` proto using the dictionary above
  return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
for parsed_record in parsed_dataset.take(1):
  print((parsed_record))