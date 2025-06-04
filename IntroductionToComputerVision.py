import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callback = myCallback()

# Accessing the data
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Normalizing the image
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the network by fitting the training images to the training labels over five epochs
model.fit(training_images, training_labels, epochs=50, callbacks=[callback])

model.evaluate(test_images, test_labels)
# Exploring the model output
classifications = model.predict(test_images)
print(classifications[4])
print(test_labels[4])