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

test = False
load = False
path = 'data\PESQ_DB.xlsx'
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
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, diff)
    plt.show()
    plt.savefig(string + '\\figure.png',bbox_inches='tight')
    return

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


def __main__():

    print("Pre-processing data ...")
    trainig, test = split_data(path)
    trainig_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1],trainig))))
    trainig_data = list(map(lambda x: x[1:],trainig))
    test_labels = list(it.chain.from_iterable(list(map(lambda x: x[:1],test))))
    test_data = list(map(lambda x: x[1:],test))

    trainig_data = np.array(trainig_data)
    trainig_labels = np.array(trainig_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if test:
        model = explore_models(arhitecture, trainig_data, trainig_labels, epochs)

    for i in [10,20]:
        for j in [10,20]:
            arhitecture = [i, j]
            string = 'models\model' + '-' + str(i) + '-' + str(j)
            os.makedirs(string)
            model = explore_models(arhitecture, trainig_data, trainig_labels, epochs)

            test_loss, test_acc = model.evaluate(test_data, test_labels)
            print(test_acc)
            print("Evaluating Model ... Done")
            own_evaluate(model, test_data, test_labels, string)
            model.save(string + '')

__main__()