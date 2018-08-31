<<<<<<< HEAD
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from cachetools import cached, TTLCache
import mdct
from scipy.io import wavfile

path1 = 'data\Female1_Seq4_20.wav'
path2 = 'data\Female1_Seq4_20_deg.wav'
cache = TTLCache(maxsize=100, ttl=300)

def main():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    # random.shuffle(data)

    train_data = data[800:]
    train_labels = labels[800:]
    test_data = data[:200]
    test_labels = labels[:200]
    #
    # cross_validation_data = train_data[:200]
    # cross_validation_labels = train_labels[:200]
    # train_data = train_data[600:]
    # train_labels = train_labels[600:]

    for i in range(10):
        for j in range(10):
            model = keras.Sequential()
            model.add(keras.layers.Dense(i, activation='relu'))
            model.add(keras.layers.Dense(j, activation='relu'))
            model.add(keras.layers.Dense(10, activation='softmax'))

            model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                          loss='mse',
                          metrics=['mse'])

            model.fit(train_data, train_labels, epochs=10, batch_size=32)

            print("\n\nPentru i = {} si j = {}:".format(i,j))
            model.evaluate(test_data, test_labels, batch_size=32)


def try_cache():
    import time
    import datetime

    @cached(cache)
    def get_candy_price(candy_id):
        # let's use a sleep to simulate the time your function spends trying to connect to
        # the web service, 5 seconds will be enough.
        time.sleep(0.1)

        # let's pretend that the price returned by the web service is $1 for candies with a
        # odd candy_id and $1,5 for candies with a even candy_id

        price = 1.5 if candy_id % 2 == 0 else 1

        return (datetime.datetime.now().strftime("%c"), price)

    # now, let's simulate 20 customers in your show.
    # They are asking for candy with id 2 and candy with id 3...
    for i in range(0, 20):
        print(get_candy_price(2))
        print(get_candy_price(3))


def try_mdct(path1, path2):
    fs, data = wavfile.read(path1)
    fs2, data2 = wavfile.read(path2)
    print(fs)
    print(fs2)
    print(data)
    print(data2)
    data_mdct = mdct.mdct(data)
    data_mdct2 = mdct.mdct(data2)
    diff = np.array(data_mdct) - np.array(data_mdct2)
    print(np.shape(data_mdct))
    print(np.shape(data_mdct2))
    plt.figure(1)
    plt.subplot(511)
    plt.plot(data)
    plt.subplot(512)
    plt.plot(data_mdct)
    plt.subplot(513)
    plt.plot(data2)
    plt.subplot(514)
    plt.plot(data_mdct2)
    plt.subplot(515)
    plt.plot(diff)
    plt.show()

try_mdct(path1, path2)

# model = keras.Sequential()
# model.add(keras.layers.Dense(10, activation='relu'))
# model.add(keras.layers.Dense(10, activation='relu'))
# model.add(keras.layers.Dense(1))
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#               loss='mse',
#               metrics=['mse'])
#
# model.load_weights('models\model-10-10\model.h5')
=======
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from cachetools import cached, TTLCache

cache = TTLCache(maxsize=100, ttl=300)

def main():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    # random.shuffle(data)

    train_data = data[800:]
    train_labels = labels[800:]
    test_data = data[:200]
    test_labels = labels[:200]
    #
    # cross_validation_data = train_data[:200]
    # cross_validation_labels = train_labels[:200]
    # train_data = train_data[600:]
    # train_labels = train_labels[600:]

    for i in range(10):
        for j in range(10):
            model = keras.Sequential()
            model.add(keras.layers.Dense(i, activation='relu'))
            model.add(keras.layers.Dense(j, activation='relu'))
            model.add(keras.layers.Dense(10, activation='softmax'))

            model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                          loss='mse',
                          metrics=['mse'])

            model.fit(train_data, train_labels, epochs=10, batch_size=32)

            print("\n\nPentru i = {} si j = {}:".format(i,j))
            model.evaluate(test_data, test_labels, batch_size=32)


def try_cache():
    import time
    import datetime

    @cached(cache)
    def get_candy_price(candy_id):
        # let's use a sleep to simulate the time your function spends trying to connect to
        # the web service, 5 seconds will be enough.
        time.sleep(0.1)

        # let's pretend that the price returned by the web service is $1 for candies with a
        # odd candy_id and $1,5 for candies with a even candy_id

        price = 1.5 if candy_id % 2 == 0 else 1

        return (datetime.datetime.now().strftime("%c"), price)

    # now, let's simulate 20 customers in your show.
    # They are asking for candy with id 2 and candy with id 3...
    for i in range(0, 20):
        print(get_candy_price(2))
        print(get_candy_price(3))

try_cache()
>>>>>>> d9ac9156e476000b8d9f7ed97955090d3cee208e
