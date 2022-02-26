import math

import numpy
import tensorflow as tf
import numpy as np
import csv


def create_model(input_shape, dp, ml, lr):  # dp =dropout rate, ml = middle layers, lr = learning rate
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dp))
    for l_i in range(ml):
        model.add(tf.keras.layers.Dense(20, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dp))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dp))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.BinaryAccuracy())
    return model


def create_validation(x_train, y_train, val_size, i):
    val_start_index = int(i * val_size)
    val_end_index = int((i + 1) * val_size)
    x_validation = x_train[val_start_index:val_end_index]
    y_validation = y_train[val_start_index:val_end_index]
    x_train = numpy.concatenate((x_train[0:val_start_index], x_train[val_end_index:]), axis=0)
    y_train = numpy.concatenate((y_train[0:val_start_index], y_train[val_end_index:]), axis=0)
    return x_train, y_train, x_validation, y_validation


def main():
    data = np.load(f"Extracted_Features/train_halved.npz")
    x_train_data, y_train_data = data["x_train"], data["y_train"]
    print(x_train_data.shape[1])
    print(y_train_data)

    if np.isnan(x_train_data).any():
        print("nan error")
        exit()
    print(f"input shape: {x_train_data.shape}")
    # network parameters
    epochs = 200
    checkpoints = 50  # including end checkpoint
    batch_size = 64
    val_folds = 7
    dropout_arr = [0.0, 0.2, 0.4]
    middle_layers_arr = [1, 2, 3]
    learning_rate_arr = [0.01, 0.001, 0.0001]

    val_folds = 7
    dropout_arr = [0.0]
    middle_layers_arr = [1]
    learning_rate_arr = [0.01]

    # empty lists for data
    train_acc = np.empty(shape=[val_folds, checkpoints])
    validation_acc = np.empty(shape=[val_folds, checkpoints])
    results = []
    # check if the size of each fold is an integer
    val_size = x_train_data.shape[0] / val_folds
    print(f"validation size: {val_size}")
    if int(val_size) != val_size:
        print("validation split error")
        exit()

    # for each combination in the grid search
    for (dropout_rate, middle_layers, learning_rate) in [(dropout_rate, middle_layers, learning_rate)
                                                         for dropout_rate in dropout_arr
                                                         for middle_layers in middle_layers_arr
                                                         for learning_rate in learning_rate_arr]:
        print(f"dp: {dropout_rate}, ml: {middle_layers}, lr: {learning_rate}")
        for i in range(val_folds):
            # cross validation
            print(f"fold: {i + 1}")
            x_train, y_train, x_validation, y_validation = create_validation(x_train_data, y_train_data, val_size, i)
            model = create_model(x_train.shape[1], dropout_rate, middle_layers, learning_rate)
            for j in range(checkpoints):
                model.fit(x_train, y_train.T, verbose=1, epochs=int(epochs / checkpoints), batch_size=batch_size)
                train_acc[i][j] = (model.evaluate(x_train, y_train, verbose=0)[1])
                validation_acc[i][j] = (model.evaluate(x_validation, y_validation, verbose=0)[1])

        train_acc_avg = numpy.average(train_acc, axis=0)
        validation_acc_avg = numpy.average(validation_acc, axis=0)
        print(f"train acc: {train_acc_avg[-1]} \n val acc: "
              f"{validation_acc_avg[-1]}")
        # creates the result lists, that are turned into a csv file later
        results.append(["training", dropout_rate, middle_layers, learning_rate] + train_acc_avg.tolist())
        results.append(["validation", dropout_rate, middle_layers, learning_rate] + validation_acc_avg.tolist())

    with open(f"classifier_results/halved_results.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile, "excel")
        writer.writerow(["type", "dropout", "middle layers", "learning rate"] + [f"{x * epochs / checkpoints} epochs"
                                                                                 for x in range(1, checkpoints + 1)])
        writer.writerows(results)
