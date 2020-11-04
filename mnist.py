# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:44:14 2020

@author: aycaburcu
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
#mnist veri setini yükledik.
#x_train bizim giriş veri setimiz
#x_train içinde görüntüler var y_train içinde ise hedefler.

(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Normalizasyon işlemleri
x_train=x_train.reshape((60000,28*28))
x_train=x_train.astype('float32')/255
x_test=x_test.reshape((10000,28*28))
x_test=x_test.astype('float32')/255

print(x_train.shape)

model=models.Sequential()
#ağın yapısını tanımlicaz


#giris katmanı
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu',input_shape=(28*28,)))
#burada 28*28 yerine 784 yazılabilir

#buraya bir katman daha ekliyoruz
model.add(layers.Dense(16,activation='relu'))
#son çıkış katmanı 1den 9a'a kadar rakamlar olucak çıkışımızda
model.add(layers.Dense(10,activation='softmax'))

#stochastic gradient descent kullanıyoruz.
##multi classification olduğu için categorical cross entropy kullanılıyor.
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model eğitilcek
model.fit(x_train,y_train,epochs=100,batch_size=64)

#sparse bu işlemi otomatik yapıyor
#mesela sayı 5 ise 5'in olduğu konuma 1 koyup diğerlerine 0 koyuyor
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

y_train[0]

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


y_train.shape

y_train[5,:]
#eğittiğimiz ağın performansını kontrol edicez şimdi
model.evaluate(x_test,y_test)
#bu bize loss ve acc döndürdü

#bu bize kullanacağımız model ne içindeki katmanları ne içinde ne var bunları verir.
model.summary()
#girişimiz kaç elemanlı=784, kaç hücre var=16 
#her hücreden girişteki elemanlara bağlantı var bu bağlantıların her birinde weight var
#784x16=12544 16 tanede her hücre için bias var 12544+16=12560
784*16+16
#ikinci katmanın giriş sayısı bir önceki katmanın çıkış sayısı
#16x16+16=
#son katmanda da 16x10+10=170
total_params=(784*16+16)+(16*16+16)+(16*10+10)
total_params


test_loss,test_acc=model.evaluate(x_test,y_test)
test_loss

w=model.get_weights()
w
#ben bu weightleri kaydedersem daha sonradan kullanabilirim
model.save('mnist.h5')
#boyutu biraz yüksek çünkü baya bir parametre var

model.summary()
model.save_weights('weightsmnist.h5')
from keras import layers
model=models.load_model('mnist.h5')
model.summary()



import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=mnist.load_data()
test_goruntu=x_test[0].reshape(1,28*28).astype('float32')/255
test_goruntu
y=model.predict(test_goruntu)
y
from numpy import argmax
rakam=argmax(y)

rakam=x_test[1000,:,:]
plt.imshow(rakam)

import numpy as np
inx=np.random.randint(0,10000)
rakam=x_test[inx,:,:]
plt.imshow(rakam)

y=model.predict(rakam.reshape(1,784)/255)
tahmin_sonucu=np.argmax(y)
print("beklenen_sonuc",y_test[inx])
print("tahmin_sonucu=",tahmin_sonucu)




