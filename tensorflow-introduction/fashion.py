import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
keras = tf.keras

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize image data to be from 0 - 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0


def train():
    # Flatten transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) 
    # to a one-dimensional array (of 28 * 28 = 784 pixels).

    # First Dense layer  has 128 nodes (neurons)

    # Last Dense layer returns logits array of length 10, each node containing a score that indicates
    # the  current image belongs to one of the 10 classes
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Optimizer: how the model is updated between epochs
    # Loss function: measures how accurate the model is during training, which "steers" the model in the right direction
    # Metrics: used to monitor the training and testing steps, "accuracy" is the fraction of the imagees that are correctly classified
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=30)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])

    return predictions

def plot_image(i, predictions_array, true_label, img): 
    true_label, img = true_label[i], img[i] 
    plt.grid(False) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(img, cmap=plt.cm.binary) 
    predicted_label = np.argmax(predictions_array) 
    if predicted_label == true_label: 
        color = 'blue' 
    else: 
        color = 'red' 
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color) 
    
def plot_value_array(i, predictions_array, true_label): 
    true_label = true_label[i] 
    plt.grid(False) 
    plt.xticks(range(10)) 
    plt.yticks([]) 
    thisplot = plt.bar(range(10), predictions_array, color="#777777") 
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array) 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue') 

def plot(predictions):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[np.argmax(predictions[i])])

    plt.show()

def test_random_images(num_images, predictions):
    for i in range(num_images):
        randint = random.randint(0,10000)
        plot_image(randint, predictions[randint], test_labels, test_images)
        plt.show()