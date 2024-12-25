import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import keras

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten_digit_recognition/handwritten.keras')

model = keras.models.load_model('handwritten_digit_recognition/handwritten.keras')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f"handwritten_digit_recognition/images_data/{image_number}.png"):
    try:
        img = cv.imread(f"handwritten_digit_recognition/images_data/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1