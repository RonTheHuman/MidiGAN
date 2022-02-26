import tensorflow as tf
from tensorflow import keras
import numpy as np
import notearr


def create_generator(noise_size, droupout_rate, weight_reg):
    generator = keras.Sequential(keras.layers.InputLayer(input_shape=noise_size))
    for size in [150, 200, 200, 200, 250]:
        generator.add(keras.layers.Dense(size, activation='tanh',
                                         kernel_regularizer=tf.keras.regularizers.L2(weight_reg)))
        generator.add(keras.layers.Dropout(rate=droupout_rate))
    generator.add(keras.layers.Dense(300))
    return generator


def create_discriminator(melody_arr_length):
    classifier = keras.Sequential(keras.layers.InputLayer(input_shape=melody_arr_length))
    leaky_relu = tf.keras.layers.LeakyReLU()
    for size in [200, 100, 50, 50, 20]:
        classifier.add(keras.layers.Dense(size, activation=leaky_relu))
    classifier.add(keras.layers.Dense(1))
    return classifier


def discriminator_loss(real_out, fake_out):
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(np.ones_like(real_out), real_out) + cross_entropy(np.zeros_like(fake_out), fake_out)


def generator_loss(fake_out):
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(np.ones_like(fake_out), fake_out)


def main():
    folder = "title"

    batch_size = 16
    noise_size = 100
    melody_arr_length = 300
    epochs = 256
    melodies_data = np.load("melody_arrays/title.npy")
    melodies_data = tf.data.Dataset.from_tensor_slices(melodies_data).batch(batch_size)

    generator = create_generator(noise_size, 0.2, 0)
    discriminator = create_discriminator(melody_arr_length)
    generator_optimizer = keras.optimizers.Adam(2*1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)

    for i in range(epochs):
        print("epoch: " + f"{i + 1}")
        for j, real_melodies in enumerate(melodies_data):
            noise = np.random.normal(size=[batch_size, noise_size])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_melodies = generator(noise)

                disc_real_out = discriminator(real_melodies)
                disc_fake_out = discriminator(fake_melodies)

                disc_loss = discriminator_loss(disc_real_out, disc_fake_out)
                gen_loss = generator_loss(disc_fake_out)

            gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            if j == 0 or j == 8:
                print(f"on real: {np.average(tf.math.sigmoid(discriminator(real_melodies)).numpy())}")
                print(f"on fake: {np.average(tf.math.sigmoid(discriminator(generator(noise))).numpy())}")
                print(f"disc loss: {disc_loss}")
                print(f"gen loss: {gen_loss}\n\n")

            generator_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    noise = np.random.normal(size=[4, noise_size])
    #print(noise)
    generated_melodies = generator(noise)
    generated_melodies = generated_melodies.numpy().astype('int')
    with open(f"music_from_arr/{folder}/{folder}.txt", "w") as file:
        file.write(repr(generated_melodies.tolist()))
    midis = notearr.note_arrs_to_midifile_arr_norm(generated_melodies, 48, 500000)
    notearr.save_midi(midis, f"music_from_arr/{folder}/{folder}")


if __name__ == '__main__':
    main()
