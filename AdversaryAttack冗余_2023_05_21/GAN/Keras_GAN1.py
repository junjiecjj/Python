#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:45:13 2023

@author: jack
"""

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential

from keras.layers import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#from google.colab import drive
#内存分析工具
from memory_profiler import profile
import objgraph

# Load the dataset
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5   # 规范化到(-1,1)
      
    # Convert shape from (60000, 28, 28) to (60000, 784)
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train)

#X_train, y_train = load_data()



def build_generator():
    model = Sequential()

    model.add(Dense(units=256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=784, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model


#generator = build_generator()


def build_discriminator():
    model = Sequential()

    model.add(Dense(units=1024 ,input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

#discriminator = build_discriminator()


@profile
def build_GAN(discriminator, generator):
    discriminator.trainable = False
    GAN_input = Input(shape=(100,))
    x = generator(GAN_input)
    GAN_output = discriminator(x)
    GAN = Model(inputs=GAN_input, outputs=GAN_output)
    GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return GAN

#GAN = build_GAN(discriminator, generator)
#print(f"GAN = \n{GAN.summary()}")

def draw_images(generator, epoch, examples=25, dim=(5,5), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    fig = plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Generated_images_%d.png' %epoch)
    plt.show()
    plt.close(fig)


@profile
def train_GAN(epochs=1, K = 1, batch_size=128, g_dim = 100):
    
    #Loading the data
    X_train, y_train = load_data()
    
    # Creating GAN
    generator= build_generator()
    discriminator= build_discriminator()
    GAN = build_GAN(discriminator, generator)

    
    for i in range(1, epochs+1):
        print("Epoch %d" %i)
        g_loss = 0
        d_loss = 0
        for _ in  range(K):
            # Generate fake images from random noiset
            noise= np.random.normal(0,1, (batch_size, g_dim))
            fake_images = generator.predict(noise)
          
            # Select a random batch of real images from MNIST
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

            # Labels for fake and real images           
            label_fake = np.zeros(batch_size)
            label_real = np.ones(batch_size) 

            # Concatenate fake and real images 
            X = np.concatenate([fake_images, real_images])
            y = np.concatenate([label_fake, label_real])

            # Train the discriminator
            discriminator.trainable = True
            d_loss += discriminator.train_on_batch(X, y)

            # Train the generator/chained GAN model (with frozen weights in discriminator) 
            discriminator.trainable = False
            GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
            g_loss += GAN.train_on_batch(noise, label_real)
        G_loss.append(g_loss)
        D_loss.append(d_loss)
            # Draw generated images every 15 epoches     
        # if i == 1 or i % 10 == 0:
        #     draw_images(generator, i)


G_loss = []
D_loss = []

train_GAN(epochs = 400, K = 128, batch_size = 128, g_dim = 100)























