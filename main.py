import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 20
#print(tf.__version__)

class_names = ['T-Shirt/top', 'Trousers', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# print('Training data: ',train_images.shape,train_labels.shape)
# print('Test data: ',test_images.shape,test_labels.shape)

def show_training_image (index):
    img_label = str(train_labels[index]) + ' (' + class_names[train_labels[index]]+ ')'
    plt.figure()
    plt.title('Image labe ' + img_label)
    plt.imshow(train_images[index], cmap = 'gray')
    plt.colorbar()
    plt.show()
#
# img_index=100
# show_training_image(img_index)

train_images = train_images/250.0
test_images = test_images/250.0

#show_training_image(img_index)
# print('Input Shape', train_images.shape)
# print()
# print(model.summary())

model = tf.keras.models.load_model('model.h5')
model.compile (optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_hist = model.fit(train_images,train_labels, EPOCHS)

def plot_acc(hist):
    plt.title("Accuracy History")
    plt.plot(hist.history['accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    plt.savefig("accuracy")

def plot_loss(hist):
    plt.title("Loss History")
    plt.plot(hist.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
    plt.savefig("loss")

plot_acc(train_hist)
plot_loss(train_hist)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose =0)
print('Max training accuracy: ',max(train_hist.history['accuracy']), ' Test accuracy: ', test_acc)