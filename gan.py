import tensorflow as tf
from tensorflow import keras
import numpy as np
import notearr
import matplotlib.pyplot as plt
from time import time


def create_generator(noise_size, droupout_rate, kernel_size, filters_arr, strides_arr):
    generator = keras.Sequential(keras.layers.InputLayer(input_shape=noise_size))
    for filters, strides in zip(filters_arr, strides_arr):
        generator.add(keras.layers.Conv1DTranspose(filters, kernel_size,
                                                   strides=strides, padding='same', activation='tanh'))
        generator.add(keras.layers.Dropout(rate=droupout_rate))
    generator.add(keras.layers.Conv1DTranspose(3, 5, strides=1, padding='same'))
    print(generator.summary())
    return generator


def create_discriminator(melody_arr_length, kernel_size, filters_arr, strides_arr):
    classifier = keras.Sequential(keras.layers.InputLayer(input_shape=(melody_arr_length, 3)))
    leaky_relu = tf.keras.layers.LeakyReLU()
    for filters, strides in zip(filters_arr, strides_arr):
        classifier.add(keras.layers.Conv1D(filters, kernel_size,
                                           strides=strides, padding='same', activation=leaky_relu))
    classifier.add(keras.layers.Flatten())
    classifier.add(keras.layers.Dense(300))
    classifier.add(keras.layers.Dense(1))
    print(classifier.summary())
    return classifier


def discriminator_loss(real_out, fake_out):
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(np.ones_like(real_out), real_out) + cross_entropy(np.zeros_like(fake_out), fake_out)


def generator_loss(fake_out):
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(np.ones_like(fake_out), fake_out)


def main():
    start_t = time()
    load = 1
    generate = 1
    batch_size = 512
    noise_size = [10, 30]
    melody_arr_length = 100
    epochs = 4096
    checkpoints = (16, 50, 100, 300, 700, 1024, 2048, 2500, 3000, 3500, 4096)
    dataset = "title"
    model_name = "conv_gan_1.1"
    run = 4
    save_dir = f"gan_results/{model_name}/{dataset}/folder/run{run}"

    generator = create_generator(noise_size, 0.5, 6, (70, 70, 70, 70), (5, 2, 1, 1))
    discriminator = create_discriminator(melody_arr_length, 6, (50, 50, 50, 50), (5, 2, 1, 1))
    exit()
    epochs_to_load = (16, 50, 100, 300, 700, 1024, 2048, 2500, 3000, 3500, 4096)
    if load and not generate:
        start_epochs = epochs_to_load[0]
        discriminator.load_weights(f"{save_dir.replace('folder', 'disc')}_epoch{start_epochs}.h5")
        generator.load_weights(f"{save_dir.replace('folder', 'gen')}_epoch{start_epochs}.h5")
        print("Loaded trained model")

    midi_to_gen = 8
    if load and generate:
        for etl in epochs_to_load:
            generator.load_weights(f"{save_dir.replace('folder', 'gen')}_epoch{etl}.h5")

            noise = np.random.normal(size=[midi_to_gen] + noise_size)
            generated_melodies = generator(noise, training=False)
            generated_melodies = generated_melodies.numpy().flatten().reshape(midi_to_gen, 300)
            # with open(f"models/{dataset}_conv.txt", "w") as file:
            #     file.write(repr(np.around(generated_melodies).astype('int').tolist()))
            midis = notearr.note_arrs_to_midifile_arr_norm(np.around(generated_melodies).astype('int').tolist(),
                                                           48, 500000)
            notearr.save_midi(midis, f"{save_dir.replace('folder', 'midi')}_epoch{etl}")
            print("Generated from loaded model")
        exit()

    generator_optimizer = keras.optimizers.Adam(2*1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)
    melodies_data = np.load(f"melody_arrays/{dataset}.npy")
    melodies_data = np.array([x.reshape(100, 3) for x in melodies_data])
    melodies_data = tf.data.Dataset.from_tensor_slices(melodies_data).batch(batch_size)

    acc_real = []
    acc_fake = []
    for i in range(epochs):
        print("epoch: " + f"{i + 1}")
        batch_start = 1
        for real_melodies in melodies_data:
            # training step
            noise = np.random.normal(size=[batch_size] + noise_size)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_melodies = generator(noise)

                disc_real_out = discriminator(real_melodies)
                disc_fake_out = discriminator(fake_melodies)

                disc_loss = discriminator_loss(disc_real_out, disc_fake_out)
                gen_loss = generator_loss(disc_fake_out)

            gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            if batch_start:
                acc_real.append(np.average(tf.math.sigmoid(disc_real_out).numpy()))
                acc_fake.append(np.average(tf.math.sigmoid(disc_fake_out).numpy()))
                print(f"on real: {acc_real[-1]}")
                print(f"on fake: {acc_fake[-1]}")
                print(f"disc loss: {disc_loss}")
                print(f"gen loss: {gen_loss}\n\n")
                batch_start = 0

            generator_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

        if i + 1 in checkpoints:
            print("Saved model\n")
            discriminator.save_weights(f"{save_dir.replace('folder', 'disc')}_epoch{i + 1}.h5")
            generator.save_weights(f"{save_dir.replace('folder', 'gen')}_epoch{i + 1}.h5")

    discriminator.save_weights(f"{save_dir.replace('folder', 'disc')}_epoch{i + 1}.h5")
    generator.save_weights(f"{save_dir.replace('folder', 'gen')}_epoch{i + 1}.h5")
    print("Saved final model")
    print(time() - start_t)
    plt.plot(acc_real, label="acc on real")
    plt.plot(acc_fake, label="acc on generated")
    plt.legend()
    plt.savefig(save_dir.replace("folder", "plots"))


if __name__ == '__main__':
    main()
