from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random

pokemon_train = []

start_string = "Spr_2g_"
data_ex = ".npy"

fig, axs = plt.subplots(1,1)


red_train = []


def show_pokemon(images, number):
    axs.cla()
    axs.imshow(images[number].reshape((56, 56, 4)), cmap=plt.get_cmap('gray'))
    plt.show()

model = load_model("first_rerun_of_colored_pokemon_autoencoder")

for i in range(251):
    image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
    pokemon_train.append(np.load(image_string))

pokemon_train = np.array(pokemon_train)
pokemon_train = pokemon_train.reshape((len(pokemon_train), np.prod(pokemon_train.shape[1:])))

predicted = model.predict(pokemon_train)
show_pokemon(predicted, 10)
exit()


random_index = np.random.rand(10000, 128)

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



#x = Conv2D(126, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(126, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(126, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)

autoencoder.fit(pokemon_train, pokemon_train,
                epochs=5000,
                batch_size=32,
                shuffle=True,
                validation_data=(pokemon_train, pokemon_train))

test_run = autoencoder.predict(pokemon_train)
show_pokemon(test_run, 0)
show_pokemon(test_run, 1)
show_pokemon(test_run, 2)
autoencoder.save("first_rerun_of_colored_pokemon_autoencoder")