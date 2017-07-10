# https://github.com/rickiepark/tfk-notebooks/tree/master/urban-sound-classification
# https://tensorflow.blog/2016/11/06/urban-sound-classification/
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]
sound_data = np.load('../../data/UrbanSound_npy/urban_sound_train.npz')
X_data = sound_data['X']
y_data = sound_data['y']
groups = sound_data['groups']
X_data.shape, y_data.shape, groups.shape

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in gss.split(X_data, y_data, groups=groups):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    groups_train = groups[train_idx]

    X_val = X_data[val_idx]
    y_val = y_data[val_idx]
    groups_test = groups[val_idx]

len(X_train), len(X_val), len(y_train), len(y_val)

# X_train = sound_data['X'][:-500]
# y_train = sound_data['y'][:-500]
# X_val = sound_data['X'][-500:]
# y_val = sound_data['y'][-500:]
# groups = sound_data['groups']


sound_data = np.load('../../data/UrbanSound_npy/urban_sound_test.npz')
X_data = sound_data['X']
y_data = sound_data['y']


training_epochs = 50
n_dim = 193
n_classes = 10
learning_rate = 0.01

n_hidden_units_one = 280
n_hidden_units_two = 300
n_hidden_units_three = 100
sd = 1 / np.sqrt(n_dim)


g1 = tf.Graph()
with g1.as_default():
    # X = tf.placeholder(tf.float32, [None, n_dim])
    # Y = tf.placeholder(tf.float32, [None, n_classes])
    #
    # c1 = tf.layers.conv2d(tf.reshape(X, [-1, 1, n_dim, 1]), 50, (1, 5), padding='same',
    #                       activation=tf.nn.sigmoid, name="c1")
    # p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[1, 2], strides=2)
    # c2 = tf.layers.conv2d(tf.reshape(p1, [-1, 1, 96, 50]), 100, (1, 5), padding='same',
    #                       activation=tf.nn.sigmoid, name="c2")
    # p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[1, 2], strides=2)
    #
    # h_p = tf.reshape(p2, [-1, 48*100])
    #
    # h_1 = tf.layers.dense(inputs=h_p, units=1000, activation=tf.nn.sigmoid,
    #                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
    #
    # y_hat = tf.layers.dense(inputs=h_1, units=n_classes,
    #                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                         name="h4")
    #
    #
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_hat))
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #
    # correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(Y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    ###########################################
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
    h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
    h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

    W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
    b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
    h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

    W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
    y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

    init = tf.initialize_all_variables()

    ###########################################
    #
    # X = tf.placeholder(tf.float32, [None, n_dim])
    # Y = tf.placeholder(tf.float32, [None, n_classes])
    #
    # W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
    # b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
    # h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)
    #
    # W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two],
    #                                    mean=0, stddev=sd))
    # b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    # h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
    #
    # W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
    # b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    # y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)
    #
    # init = tf.initialize_all_variables()

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))


cost_history = []
with tf.Session(graph=g1) as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        ac, cost = sess.run([accuracy, cost_function], feed_dict={X: X_train, Y: y_train})
        cost_history.append(cost)
        print("Epoch: ", epoch, " Training Loss: ", cost, " Training Accuracy: ", ac)

    print('Validation accuracy: ', round(sess.run(accuracy, feed_dict={X: X_val, Y: y_val}), 3))
    print('Last cost: ', round(cost_history[-1], 3))

    sess.close()
plt.plot(cost_history)


# cost_history = []
# sess = tf.Session(graph=g1)
# tf.reset_default_graph()
# sess.run(init)
# for epoch in range(training_epochs):
#     _, cost = sess.run([optimizer, loss], feed_dict={X: X_data, Y: y_data})
#     cost_history.append(cost)
#
# print('Last cost: ', round(cost_history[-1], 3))
# plt.plot(cost_history)
# saver.save(sess, "model_adam.ckpt")
# sess.close()


# with tf.Session(graph=g1) as sess:
#     sess.run(init)
#     saver.restore(sess, 'model_adam.ckpt')
#     print('Test accuracy: ',round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}) , 3))
