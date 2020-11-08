#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:50:44 2020

@author: aycaburcu
"""
from keras.datasets import imdb
import numpy as np
from keras import Sequential,models,layers
from keras.datasets import mnist
import matplotlib.pyplot as plt

#Veri setini yükle ve hazırla----------------------------------
(train_data,train_labels),(test_data,test_labels)=\
imdb.load_data(num_words=10000)
print(train_data.shape)
print(test_data.shape)
print(train_data[0])
print(train_labels[0])
train_labels[100]
s=np.sort(train_data[0])
s
s.shape
s.reshape(218,1)

def  vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequences in enumerate(sequences):
        results[i,sequences]=1.
    return results
#eğitim ve test verilerini vektöre dönüştürdük
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)
x_train.shape
x_train[0,:]


#etiketleri vektöre dönüştür
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')
#Modeli Tanımla----------------------------------
model=models.Sequential()

model.add(layers.Dense(16,
                       activation='relu',
                       use_bias=True,
                       #bunu kullanırsak bias olmaz
                       input_shape=(10000,)))
#vektör formatında olduğu için 10000 dedikten sonra , koyuyoruz 
# görüntü olsa 28*28 olucak mesela renkliyse bide üstüne 28*28 çarpı bide 3 gelicek
model.add(layers.Dense(32,
                       activation='relu'))


model.add(layers.Dense(1,
                       activation='sigmoid'))
#modeli tanımladık burada summary ile özetini görelim 
model.summary()
#birinci katmanda 10000*16 tane bağlantı var +16 tane de hücrenin biası var
#modeli derle -------------------------------------------
from keras import optimizers,losses,metrics
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              #loss='binary_crossentropy',
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
              #metrics=['accuracy'])
'''
model.fit(x_train,y_train,
          epochs=5,
          batch_size=32)
'''
x_val=x_test[:5000]

y_val=y_test[:5000]

'''
history=model.fit(x_train,y_train,
          epochs=5,
          batch_size=32,
          validation_data=(x_val,y_val))
'''
history=model.fit(x_train,y_train,
          epochs=5,
          batch_size=32,
          validation_split=0.2)
#history grafiğini çizdiriyoruz
history_dict=history.history
loss_values=history_dict['loss']#loss değerleri
val_loss_values=history_dict['val_loss']#validation loss değerleri
epochs=range(1,len(loss_values)+1)
# 5 taneyse 1den 6 ya kadar götürdük hatırlarsak +1 dememizin sebebi o 


plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

#[loss,accuracy]=model.predict(x_test,y_test)
test_loss,test_acc=model.evaluate(x_test,y_test)
model.predict(x_test[100,:].reshape(1,10000))
print('test_acc',test_acc)
print('test_loss',test_loss)


























