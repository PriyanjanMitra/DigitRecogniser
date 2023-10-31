import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
model=tf.keras.models.load_model('handwritten_digits.model')
for x in range(0, 10):
    img = cv.imread(f'{x}.png')
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably:{np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
