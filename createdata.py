import pandas
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    folder = "extracted_features"
    halved = "_halved"
    # turn exel file to dataframe, f is short for features
    f_title = pandas.read_csv(f"{folder}/title{halved}_features.csv")
    f_battle = pandas.read_csv(f"{folder}/battle{halved}_features.csv")

    """
    remove file names and save feature names, because pandas removes column titles.
    pandas is used for its built in exel -> numpy functions
    """
    f_title = f_title.drop(f_title.columns[0], axis=1)
    f_battle = f_battle.drop(f_battle.columns[0], axis=1)
    # turn dataframe to numpy array
    f_title = f_title.to_numpy().astype(object)
    f_battle = f_battle.to_numpy().astype(object)
    # concatenate on vertical (list of files) axis
    data_x = np.concatenate((f_title, f_battle), axis=0).astype('float32')
    for ij in np.argwhere(np.isnan(data_x)):
        data_x[ij[0], ij[1]] = 0
    if np.isnan(data_x).any():
        print("nan in data")
        exit()
    # make class labels for each dataset
    c_title = np.ones(f_title.shape[0])
    c_battle = np.zeros(f_battle.shape[0])
    data_y = np.concatenate((c_title, c_battle), axis=0)
    print(f"data_x:\n {data_x} \n {data_x.shape} \n data_y:\n {data_y} \n {data_y.shape}")
    # Use train_test_split to handle matrix stuff. Though its probably not complicated.
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    print(f"x_train:\n{x_train} \n x_test:\n{x_test} \n y_train:\n{y_train} \n y_test:\n{y_test}")
    # save to numpy files
    np.savez(f"{folder}/train{halved}.npz", x_train=x_train, y_train=y_train)
    np.savez(f"{folder}/test{halved}.npz", x_test=x_test, y_test=y_test)
