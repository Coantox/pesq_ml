import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DataPESQ import DataPESQ
import itertools as it
import os, sys, getopt

in_dir = ['data\PESQ_DB.xlsx', 'data\PESQ_DB_Male1_Seq4.xlsx']
out_dir = 'results'
cache_path = 'data/cache'
load_model = False
arg_list = ['--load-model', '--out-dir', '--in-dir', '--test-model', '--view-only']
arhitecture = [30, 30]


def plot_results(model, test_data, test_labels, rate=0.2):
    test_labels_array = list(map(lambda x: [x], test_labels))
    x = list(map(lambda x: x[0], test_data))
    y = list(map(lambda x: x[1], test_data))
    true_results = np.concatenate((test_data, test_labels_array), axis=1)
    prediction = model.predict(test_data)
    prediction_vector = np.concatenate((prediction),axis=None)
    model_results = np.concatenate((test_data, prediction), axis=1)
    diff = test_labels - prediction_vector
    index_good = [i for i in range(len(diff)) if diff[i] <= rate]
    index_bad = [i for i in range(len(diff)) if diff[i] > rate]
    print(len(index_good)/len(diff))
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter([x[i] for i in index_good], [y[i] for i in index_good], [diff[i] for i in index_good])
    ax.scatter([x[i] for i in index_bad], [y[i] for i in index_bad], [diff[i] for i in index_bad], color='r')
    plt.show()
    return


def view_last_model():
    raise NotImplementedError


def test_model(model, test_data, test_labels):
    test_loss , test_mse, test_mae, test_mape = model.evaluate(test_data, test_labels)
    print(test_loss, test_mse, test_mae, test_mape)


def get_data(path_list, training_size=0.8):
    data = np.array([[1,1,1]])

    # extract data from files
    for path in path_list:
        data_obj = DataPESQ(path)
        data_from_ob = np.array(data_obj.get_data())
        data = np.concatenate((data, data_from_ob), axis=0)
        # np.random.shuffle(data)
    data = data[1:]
    np.random.shuffle(data)

    # normalize data
    data_tr = np.transpose(data)
    for i in [0, 1, 2]:
        xmin = data_tr[i].min()
        xmax = data_tr[i].max()
        print(xmin, xmax)
        std = xmax - xmin
        for elem in data_tr[i]:
            elem = (elem - xmin) / std
    data = np.transpose(data_tr)
    # split data
    sample_test = round(data.shape[0]*(1 - training_size))
    training, test = data[sample_test:], data[:sample_test]
    training_labels = np.array(list(it.chain.from_iterable(list(map(lambda x: x[:1], training)))))
    training_data = np.array(list(map(lambda x: x[1:], training)))
    test_labels = np.array(list(it.chain.from_iterable(list(map(lambda x: x[:1], test)))))
    test_data = np.array(list(map(lambda x: x[1:], test)))
    return training_data, training_labels, test_data, test_labels


def create_model(lst):
    model = keras.Sequential()

    for x in lst:
        model.add(keras.layers.Dense(x, activation='relu'))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='mse',
                  metrics=['mse', 'mae', 'mape'])
    return model


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "  ", arg_list)
    except getopt.GetoptError:
        print()
        sys.exit(2)

    # Look for options
    for opt, arg in opts:
        if opt in ('-l', '--load-model'):
            print("model")
            global load_model
            load_model = True
        if opt == '--out-dir':
            global out_dir
            out_dir = arg
        if opt == '--in-dir':
            global in_dir
            in_dir = arg

    # Look for commands
    for opt, arg in opts:
        if opt in ('-t', '--test-model'):
            print("testing model")

            return
        if opt in ('-v', '--view-only'):
            print("view results for last model tested")
            return

    training_data, training_labels, test_data, test_labels = get_data(in_dir)
    if load_model:
        print("load_model")
        model = create_model(arhitecture)
        model.fit(np.array([[1, 1]]), np.array([[1]]))
        model.summary()
        model.load_weights('my_model.h5')
    else:
        model = create_model(arhitecture)
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                  write_graph=True, write_images=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            './checkpoints/checkpoints', verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=5)
        print(model)
        model.fit(training_data, training_labels, epochs=100, validation_split=0.3, callbacks=[tb_callback])
        model.save('my_model.h5')

    test_model(model, test_data, test_labels)
    plot_results(model, test_data, test_labels)


if __name__ == "__main__":
    main(sys.argv[1:])