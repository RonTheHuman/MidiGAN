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
    x_train = np.concatenate((x_train[0:val_start_index], x_train[val_end_index:]), axis=0)
    y_train = np.concatenate((y_train[0:val_start_index], y_train[val_end_index:]), axis=0)
    return x_train, y_train, x_validation, y_validation


def main():
    # load data, npz file contains two numpy arrays
    train_data = np.load(f"Extracted_Features/train.npz")
    x_train, y_train = train_data["x_train"], train_data["y_train"]
    # exit before a crash happens if any feature is none
    if np.isnan(x_train).any():
        print("nan error")
        exit()
    print(f"input shape: {x_train.shape}")
    # network parameters
    epochs = 200
    acc_checks = 50  # including end checkpoint
    batch_size = 64
    val_folds = 7
    # values for grid search
    dropout_arr = [0.0, 0.2, 0.4]
    middle_layers_arr = [1, 2, 3]
    learning_rate_arr = [0.01, 0.001, 0.0001]
    # empty lists for data
    train_acc = np.empty(shape=[val_folds, acc_checks])
    validation_acc = np.empty(shape=[val_folds, acc_checks])
    results = []
    # check if the size of each fold is an integer
    val_size = x_train.shape[0] / val_folds
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
            x_val_train, y_val_train, x_validation, y_validation = create_validation(x_train, y_train, val_size, i)
            model = create_model(x_val_train.shape[1], dropout_rate, middle_layers, learning_rate)
            for j in range(acc_checks):
                model.fit(x_val_train, y_val_train.T, verbose=1, epochs=int(epochs / acc_checks), batch_size=batch_size)
                # save acc data for graph. discovered later that there is a built in feature for it :|
                train_acc[i][j] = (model.evaluate(x_val_train, y_val_train, verbose=0)[1])
                validation_acc[i][j] = (model.evaluate(x_validation, y_validation, verbose=0)[1])
        # averages data over all validation runs
        train_acc_avg = np.average(train_acc, axis=0)
        validation_acc_avg = np.average(validation_acc, axis=0)
        print(f"train acc: {train_acc_avg[-1]} \n val acc: {validation_acc_avg[-1]}")
        # creates the result lists, that are turned into a csv file later
        results.append(["training", dropout_rate, middle_layers, learning_rate] + train_acc_avg.tolist())
        results.append(["validation", dropout_rate, middle_layers, learning_rate] + validation_acc_avg.tolist())
    # write results to csv
    with open(f"classifier_results/Data/halved_results.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile, "excel")
        writer.writerow(["type", "dropout", "middle layers", "learning rate"] + [f"{x * epochs / acc_checks} epochs"
                                                                                 for x in range(1, acc_checks + 1)])
        writer.writerows(results)
