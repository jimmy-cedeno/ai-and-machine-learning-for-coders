import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

IMAGE_SIZE = (224, 224)

def format_image(image, label):
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0
  return image, label

(raw_train, raw_validation, raw_test), metadata = tfds.load(
  'cats_vs_dogs',
  split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
  with_info = True,
  as_supervised = True
)

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print(num_examples)
print(num_classes)

BATCH_SIZE = 32
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

for image_batch, label_batch in train_batches.take(1):
  pass
image_batch.shape

module_selection = ('mobilenet-v2', 244, 1280)
handle_base, pixels, FV_SIZE = module_selection
print(handle_base)

MODULE_HANDLE = 'https://www.kaggle.com/models/google/{}/TensorFlow2/tf2-preview-feature-vector/4'.format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print('Using {} with size {} and output dimension {}'.format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                  #  input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False)

print('Building model with', MODULE_HANDLE)

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.build([None, 224, 224, 3])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



hist = model.fit(train_batches,
                 epochs=5,
                 validation_data=validation_batches)

CATS_VS_DOGS_SAVED_MODEL = 'exp_saved_model'
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
  for input_value, _ in test_batches.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

predictions = []
test_labels, test_imgs = [], []
for img, label in test_batches.take(100):
  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predictions.append(interpreter.get_tensor(output_index))
  test_labels.append(label.numpy()[0])
  test_imgs.append(img)

score = 0
for item in range(0,99):
  prediction = np.argmax(predictions[item])
  label = test_labels[item]
  if prediction==label:
    score += 1

print('Out of 100 predictions I got ' + str(score) + 'correct.')

class_names = ['cats', 'dogs']
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  img = np.squeeze(img)
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)\
  
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  
  plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]), color=color)
  
for index in range(0, 99):
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(index, predictions, test_labels, test_imgs)
  plt.show()