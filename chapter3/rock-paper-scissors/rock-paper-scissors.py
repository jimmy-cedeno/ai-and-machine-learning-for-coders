import zipfile
import urllib.request
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

training_url = "https://storage.googleapis.com/learning-datasets/rps.zip"
training_file_name = "rps.zip"
training_dir = 'rps/training/'
os.makedirs(training_dir, exist_ok=True)
urllib.request.urlretrieve(training_url, training_file_name)
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_url = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"
validation_file_name = "rps-test-set.zip"
validation_dir = 'rps/validation/'
os.makedirs(validation_dir, exist_ok=True)
urllib.request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

test_url = "https://storage.googleapis.com/learning-datasets/rps-validation.zip"
test_file_name = "rps-validation.zip"
test_dir = 'rps/test/'
os.makedirs(test_dir, exist_ok=True)
urllib.request.urlretrieve(test_url, test_file_name)
zip_ref = zipfile.ZipFile(test_file_name, 'r')
zip_ref.extractall(test_dir)
zip_ref.close()

training_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = training_datagen.flow_from_directory(
  training_dir + '/rps/',
  target_size=(150, 150),
  class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir + '/rps-test-set/',
    target_size=(150, 150),
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
  # Note the input shape is the desired size of the image 150x150 with 3 bytes color
  # This is the first convolution
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  # The second convolution
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
  # The third convolution
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
  # The fourth convolution
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  
  # Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  # 512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  # 3 output neurons for 3 classes
  tf.keras.layers.Dense(3, activation='softmax')
])



model.compile(loss= 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(
  train_generator,
  epochs=20,
  verbose=1,
  validation_data=validation_generator
)


# Test the model
test_path = "rps/test/"
labels = ["PAPER", "ROCK", "SCISSOR"]
for fn in os.listdir(test_path):
    path = os.path.join(test_path, fn)
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor, batch_size=10)

    print(fn)
    print(classes[0])

    predicted_index = np.argmax(classes[0])
    print(f'Prediction {fn}: {labels[predicted_index]}')
    # if classes[0] > 0.5:
    #     print(f'{fn} is a human')