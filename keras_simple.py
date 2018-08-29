import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DataPESQ import DataPESQ
import itertools as it
from cachetools import cached, TTLCache
import os

cache = TTLCache(maxsize=3000, ttl=9000)

test_flag = False
load = False
path_female = 'data\PESQ_DB.xlsx'
path_test = 'data\PESQ_DB_Male1_Seq4.xlsx'
path_male = 'data\PESQ_DB_Male1_Seq4.xlsx'
path_model = 'models\model-10-10\model.h5'
init_momentum = 0.9
arhitecture = [30,30]
epochs = 100


def own_evaluate(model, test_data, test_labels, string):
    test_labels_array = list(map(lambda x: [x], test_labels))
    x = list(map(lambda x: x[0], test_data))
    y = list(map(lambda x: x[1], test_data))
    # print(test_labels_array)
    true_results = np.concatenate((test_data, test_labels_array), axis=1)
    prediction = model.predict(test_data)
    prediction_vector = np.concatenate((prediction),axis=None)
    model_results = np.concatenate((test_data, prediction), axis=1)
    diff = test_labels - prediction_vector

    # diff = list(map(lambda x: np.array(x), diff))
    # diff_results = np.concatenate((test_data, diff), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(x, y, diff)
    ay = fig.add_subplot(212)
    ay.hist(diff, bins=10)
    # plt.show()
    plt.savefig(string + '\\figure.png',bbox_inches='tight')
    return


def test_data_only(path):
    dataObj = DataPESQ(path)
    data = dataObj.get_data()
    np.random.shuffle(data)
    print(np.shape(data))
    return data

@cached(cache)
def split_data(path):
    dataObj = DataPESQ(path)
    data = dataObj.get_data()
    np.random.shuffle(data)
    sample_train = round(len(data) * 0.8)
    sample_test = len(data) - sample_train
    print(len(data), sample_train, sample_test)
    training, test = data[sample_test:], data[:sample_test]
    print(np.shape(training))
    print(np.shape(test))
    return training, test

def explore_models(list, trainig_data, training_labels, epochs):
    model = keras.Sequential()

    for x in list:
        model.add(keras.layers.Dense(x,activation='relu'))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                  loss='mse',
                  metrics=['mse'])
    model.fit(trainig_data, training_labels, epochs=epochs)
    return model

def test_model(path_database, path_model):
    print("Testing model at " + path_model)
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                  loss='mse',
                  metrics=['mse'])
    # model.summary()
    # exit(0)
    model = keras.models.load_model(path_model)
    print("Getting data")
    data = test_data_only(path_database)
    test_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1], data))))
    test_data = list(map(lambda x: x[1:], data))
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    print("Getting data DONE")
    model.evaluate(test_data, test_labels)

def __main__():

    print("Pre-processing data ...")
    trainig_male, test_male = split_data(path_male)
    trainig_female, test_female = split_data(path_female)
    training = np.concatenate((trainig_male, trainig_female), axis=0)
    np.random.shuffle(training)
    test = np.concatenate((test_male, test_female), axis=0)
    np.random.shuffle(test)
    trainig_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1],training))))
    trainig_data = list(map(lambda x: x[1:],training))
    test_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1],test))))
    test_data = list(map(lambda x: x[1:],test))

    trainig_data = np.array(trainig_data)
    trainig_labels = np.array(trainig_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(log_dir='.\logs')
    ]

    print("Test = ", test_flag)
    if test_flag:
        arhitecture = [30, 20]
        model = explore_models(arhitecture, trainig_data, trainig_labels, epochs)
        # own_evaluate(model, test_data, test_labels, 'test')
        # print("Getting data")
        # data = test_data_only(path_test)
        # test_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1], data))))
        # test_data = list(map(lambda x: x[1:], data))
        # test_data = np.array(test_data)
        # test_labels = np.array(test_labels)
        # print("Getting data DONE")
        test_loss, test_acc = model.evaluate(test_data, test_labels)
        print(test_acc)
        own_evaluate(model, test_data, test_labels, 'test')
        exit(0)

    for i in range(1,51):
        for j in range(1,51):
            try:
                arhitecture = [i, j]
                string = 'models\model' + '-' + str(i) + '-' + str(j)
                os.makedirs(string)
                model = explore_models(arhitecture, trainig_data, trainig_labels, epochs)
                file = open(string + '\log.txt', 'w')
                test_loss, test_acc = model.evaluate(test_data, test_labels)
                print(test_acc)
                file.write('test accuracy')
                file.write(str(test_acc))
                file.close()
                print("Evaluating Model ... Done")
                own_evaluate(model, test_data, test_labels, string)

                # model.evaluate(test_data, test_labels)
                model.save(string + '\model.h5')
            except Exception:
                pass

__main__()
# test_model(path_test, path_model)