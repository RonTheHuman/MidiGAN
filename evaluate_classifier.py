import tensorflow as tf
import numpy as np


def create_model(input_shape, dp, ml, lr):  # dp =dropout rate, ml = middle layers, lr = learning rate
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


def main():
    halved = ""
    print(halved)
    train_data = np.load(f"Extracted_Features/train{halved}.npz")
    x_train, y_train = train_data["x_train"], train_data["y_train"]
    test_data = np.load(f"Extracted_Features/test{halved}.npz")
    x_test, y_test = test_data["x_test"], test_data["y_test"]

    val_size = int(x_train.shape[0]/7)
    x_val = x_train[0:val_size]
    y_val = y_train[0:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    print(f"{x_train.shape} \n {x_val.shape} \n {x_test.shape} \n\n "
          f"{y_train.shape} \n {y_val.shape} \n {y_test.shape} \n\n")

    batch_size = 64
    dropout = 0
    middle_layers = 1
    learning_rate = 0.01
    runs = 100
    val_acc_arr = []

    model = create_model(x_train.shape[1], dropout, middle_layers, learning_rate)
    for i in range(runs):
        tf.keras.backend.clear_session()
        history = model.fit(x_train, y_train.T, verbose=0, epochs=200, batch_size=batch_size)
        val_acc = model.evaluate(x_val, y_val.T, verbose=0)[1]
        val_acc_arr.append(val_acc)
        model.save_weights(f"classifier_results/Models/model{halved}_{i}.h5")
        print(f"saved model no {i}")
        # print(history.history["binary_accuracy"])
        print(val_acc_arr)

    best_model = max(range(len(val_acc_arr)), key=lambda x: val_acc_arr[x])  # argmax on list of accuracies
    print(f"best model: {best_model}")
    tf.keras.backend.clear_session()
    model.load_weights(f"classifier_results/Models/model{halved}_{5}.h5")
    test_acc = model.evaluate(x_test, y_test.T, verbose=0)[1]
    print(f"accuracy on test: {test_acc}")
    with open(f"classifier_results/Data/best_acc_on_test{halved}.txt", "w") as file:
        file.write(str(test_acc))
