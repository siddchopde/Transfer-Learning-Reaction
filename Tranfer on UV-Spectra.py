# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:33:25 2018

@author: t-sichop
"""

%matplotlib inline
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

#Code to scale the input data from range a,b to 0,255
def normalize(x):
    x = (x-x.min())/(x.max()-x.min())
    return x


#Load VGG16 Model with imagenet weights
model = VGG16( weights='imagenet', include_top=True,input_shape=(224, 224, 3))

input_shape = model.layers[0].output_shape[1:3]
input_shape

model.summary()



#get reference of the output of last convulational layer
transfer_layer = model.get_layer('block5_pool')
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
transfer_layer.output

#create a new sequential model
new_model = Sequential()

#add the cut vgg16 model as first part
new_model.add(conv_model)
new_model.add(Flatten())

#add new layers
new_model.add(Dense(512, activation='sigmoid'))
#new_model.add(Dense(256,input_dim=512,activation='tanh'))
new_model.add(Dense(1,kernel_initializer = 'normal'))
#new_model.add(Dropout(0.5))

optimizer1 = Adam(lr=.00005)

conv_model.trainable = False

for layer in conv_model.layers:
    layer.trainable = False

for layer in conv_model.layers:
    print(layer.name,layer.trainable)
    
#Use this block for fine tuning    
'''for layer in conv_model.layers:
   trainable = ('block5' in layer.name  or 'block4' in layer.name)
    
   layer.trainable = trainable'''
new_model.compile(loss='mean_squared_error',optimizer = optimizer1)

new_model.summary()


#load the data from METING datasets
fig=plt.figure(figsize=(1,8))
columns = 1
rows = 8
cur = 7
data_train = []
for i in range(3,11):
    name = '18/METING'+str(cur+i)+'.MAT'
    arr = 'x' + str(cur+i) + 'b'    
    a = sio.loadmat(name)[arr][:224,1:225]
    img = normalize(a)
    #load_img = np.stack((img,)*3,-1)
    load_img = np.stack((img,np.zeros((224,224)),np.zeros((224,224))),-1)

    #l_img = load_img.reshape(1,load_img.shape[0],load_img.shape[1],load_img.shape[2])
    data_train.append(load_img)
    fig.add_subplot(rows,columns,i-2)
    plt.imshow(img,cmap='hot')
    plt.show()
data_train = np.array(data_train)
Y=[175.67]*8


data_test = []
for i in range(1,3):
    name = '18/METING'+str(cur+i)+'.MAT'
    arr = 'x' + str(cur+i) + 'b'    
    a = sio.loadmat(name)[arr][:224,1:225]
    img = normalize(a)
    #load_img = np.stack((img,)*3,-1)
    load_img = np.stack((img,np.zeros((224,224)),np.zeros((224,224))),-1)

    #l_img = load_img.reshape(1,load_img.shape[0],load_img.shape[1],load_img.shape[2])
    data_test.append(load_img)
Y_test = [175.67]
data_test = np.array(data_test)
new_model.fit(data_train,Y,validation_split=0.1,epochs=1)

data_plot = []

Y_plot = [175.67]*10

cost = new_model.evaluate(data_test,Y_test)
ans = new_model.predict(data_test)
ans2 = new_model.predict(data_train)
ansk = np.concatenate((ans2,ans))
print(cost)

ansk = list(i[0] for i in ansk)
xp = [i for i in range(1,11)]
error = [abs(i-j) for i,j in zip(ansk,Y_plot)]

abc = plt.barh(xp,error)
abc[8].set_color('r')
abc[9].set_color('r')
plt.xlabel('Absoulte Difference')
plt.ylabel('Data Set')
for i in range(1,11):
    plt.text(error[i-1] - 0.1,i-0.17, str(ansk[i-1]), color='black', fontweight='bold')
plt.show()