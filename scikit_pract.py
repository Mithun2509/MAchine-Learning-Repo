# from sklearn import datasets
# from sklearn import svm

# digits=datasets.load_digits()

# clf=svm.SVC(gamma=0.001,C=100.)

# clf.fit(digits.data[:-1],digits.target[:-1])

# clf.predict(digits.data[-1:])

from tensorflow import keras
import tensorflow as tf


new_model = tf.keras.models.load_model('mnist_model.h5')

# Show the model architecture
new_model.summary()