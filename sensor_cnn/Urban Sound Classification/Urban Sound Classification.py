# http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=1
# https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
# http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2
# https://tensorflow.blog/2016/11/06/urban-sound-classification/
import glob
import os
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
#% matplotlib
#inline


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        print(fp)
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25, 60), dpi=900)
    fig = plt.figure()
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25, 60), dpi=900)
    fig = plt.figure()
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25, 60), dpi=900)
    fig = plt.figure()
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


def extract_feature(file_name):
    try:
        X, sample_rate = librosa.load(file_name)
    except:
        data, sample_rate = sf.read(file_name, dtype='float32')

        try:
            if data.shape[1] is not None:
                X = data.T[0]
        except:
            X = data

        # data = data.T
        # X = librosa.resample(data, samplerate, 22050)
        # sample_rate = 22050

    try:
        stft = np.abs(librosa.stft(X))
    except:
        print("0")
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('\\')[1].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


#
# parent_dir = '../../data/UrbanSound8K/audio/fold1/'
# sound_file_paths = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav","46669-4-0-35.wav",
#                    "89948-5-0-0.wav","40722-8-0-4.wav","106905-8-0-0.wav","108041-9-0-4.wav"]
# sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
#                "gun shot","siren","street music"]
#
# cnt = 0
# for i in sound_file_paths:
#     sound_file_paths[cnt] = parent_dir + i
#     cnt = cnt+1
#
# raw_sounds = load_sound_files(sound_file_paths)
#
# plot_waves(sound_names,raw_sounds)
# plot_specgram(sound_names,raw_sounds)
# plot_log_power_specgram(sound_names,raw_sounds)



parent_dir = '../../data/UrbanSound8K/audio/'
tr_sub_dirs = ['fold1','fold2']
ts_sub_dirs = ['fold3']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01


X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()


cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: tr_features, Y: tr_labels})
        cost_history = np.append(cost_history, cost)

        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
        y_true = sess.run(tf.argmax(ts_labels, 1))
        print('Test accuracy: ', round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))

fig = plt.figure(figsize=(10, 8))
plt.plot(cost_history)
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

# p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
# print ("F-Score:", round(f, 3))