#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Lambda, LeakyReLU, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import tensorflow as tf

import os

#%%
(X_train,y_train),(X_test,y_test) = mnist.load_data()
x_csv = pd.read_csv('C:\\Users\ABDUL BASID\Desktop\\2019\digit-recognizer\\test.csv')

X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
x_csv=x_csv.values.reshape(-1,28,28,1).astype('float32')

print(X_train.shape)
print(X_test.shape)
print(x_csv.shape)

#%%
X_train_ = X_train.reshape(X_train.shape[0], 28, 28)

fig, axis = plt.subplots(3, 3, figsize=(10, 12))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train_[i], cmap=plt.get_cmap('binary'))
    ax.set(title = "Real Number is {}".format(y_train[i]));

#%%
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)


#%%
mean = np.mean(X_train)
std = np.std(X_train)
def standardize(x):
    return (x-mean)/std


#%%
def model3():
    model=Sequential()

    model.add(Lambda(standardize,input_shape=(28,28,1)))
    
    model.add(Conv2D(32,(3,3), padding = 'Same'));   model.add(BatchNormalization());   model.add(LeakyReLU(alpha=0.25))
    model.add(Conv2D(32,(3,3), padding = 'Same'));   model.add(BatchNormalization());   model.add(LeakyReLU(alpha=0.2))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding = 'Same'));   model.add(BatchNormalization());    model.add(LeakyReLU(alpha=0.25))
    model.add(Conv2D(64,(3,3), padding = 'Same'));   model.add(BatchNormalization());    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3), padding = 'Same'));   model.add(BatchNormalization());    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3), padding = 'Same'));   model.add(BatchNormalization());    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten());    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    model.add(Dense(256));    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.15))
    
    model.add(Dense(128));    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.15))
    
    model.add(Dense(10,activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-8, momentum=0.95, nesterov=True)
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    return model

#%%
model=model3()

#%%
model.load_weights('C:\\Users\\ABDUL BASID\\Desktop\\2019\\digit-recognizer\\cp-0040.ckpt')
loss, acc = model.evaluate(X_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%
y = model.predict(x_csv, batch_size=None, verbose=1, steps=None)

#%%
print(y.shape)

#%%
y = np.argmax(y,axis = 1)
y = pd.Series(y,name="Label")

#%%
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y],axis = 1)
# Create final csv file for submission
submission.to_csv("cnn_mnist_datagen2.csv",index=False)
