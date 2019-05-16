from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential

import numpy as np
import random


pokemon_train = []

start_string = "Spr_2g_"
data_ex = ".npy"

for i in range(251):
    image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
    pokemon_train.append(np.load(image_string))

pokemon_train = np.array(pokemon_train)
pokemon_train = pokemon_train.reshape((len(pokemon_train), np.prod(pokemon_train.shape[1:])))


input_size = 56 * 56 * 4
encoding_dim = 32

input_img = Input(shape=(input_size,))

encoded = Dense(1024, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(256, activation='relu')(encoded)

decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(56 * 56 * 4, activation='sigmoid')(decoded)
 
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(pokemon_train, pokemon_train,
                epochs=5000,
                batch_size=32,
                shuffle=True,
                validation_data=(pokemon_train, pokemon_train))

def split_network(network):
	first_half_of_network = Sequential()
	first_half_of_network.add(network.layers[0])
	first_half_of_network.add(network.layers[1])
	first_half_of_network.add(network.layers[2])
	first_half_of_network.add(network.layers[3])
	first_half_of_network.compile(optimizer='adadelta', loss='mean_squared_error')

	second_half_of_network = Sequential()
	second_half_of_network.add(network.layers[4])
	second_half_of_network.add(network.layers[5])
	second_half_of_network.add(network.layers[6])
	second_half_of_network.compile(optimizer='adadelta', loss='mean_squared_error')
	return (first_half_of_network, second_half_of_network)

(encoder, decoder) =  split_network(autoencoder)

random_interop = []



critic_input_img = Input(shape=(56 * 56 * 4, ))
critic = Dense(128, activation='relu')(critic_input_img)
critic = Dense(128, activation='relu')(critic)
critic = Dense(128, activation='relu')(critic)
critic = Dense(1, activation='linear')(critic)

critic_network = Model(critic_input_img, critic)
critic_network.compile(optimizer='adadelta', loss='mean_squared_error')

for i in range(5000):
	random_interop.append(random.uniform(0, 1))

results = encoder.predict(pokemon_train)


interpolated_images = []

for i in range(5000):
	alpha = results[i % len(results)]
	beta = results[(i + 150) % len(results)]
	alpha_prime = np.multiply(alpha, 1.0 - random_interop[i])
	beta_prime = np.multiply(beta, random_interop[i])
	interpolated_images.append(np.add(alpha_prime, beta_prime))

interpolated_images = np.array(interpolated_images)
interpolated_images = interpolated_images.reshape((len(interpolated_images), np.prod(interpolated_images.shape[1:])))
decoded_images = decoder.predict(interpolated_images)

critic_network.fit(decoded_images, np.array(random_interop), epochs=5000, batch_size=32, shuffle=True, validation_data=(decoded_images, np.array(random_interop)))
np.save("random_intern_vals", random_interop)
np.save("interp_images", interpolated_images)
np.save("critic_network", critic_network)
np.save("decoded_iamges", decoded_images)