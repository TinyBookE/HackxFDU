import numpy as np
# import wave
import librosa
# from python_speech_features import mfcc
import os

"""
def read_wave(fname):
    f = wave.open(fname,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    str_data = f.readframes(nframes)

    f.close()
    wave_data = np.fromstring(str_data, dtype = np.float32).reshape(2, -1)

    #time = np.arange(0, nframes) * (1.0/framerate)

    return mfcc(wave_data[0]),mfcc(wave_data[1])
"""


def read_mp3(fname):
    y, sr = librosa.load(fname)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    chromagram = librosa.feature.chroma_cqt(y_harmonic, sr=sr)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta]).T
    while(beat_features.shape[0] < 100):
        beat_features = np.vstack((beat_features,beat_features))
    beat_features = beat_features[:100]
    print('load finished')
    return beat_features


def dense_to_one_hot(label, num_classes=6):
    num_labels = label.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + label.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, music, labels):
        assert music.shape[0] == labels.shape[0]

        self._num_examples = music.shape[0]
        self._music = music
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def music(self):
        return self._music

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._music = self._music[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._music[start:end], self._labels[start:end]


def read_data_sets(fpath, one_hot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()
    flag = True
    _labels = []
    genre_list = ["fork", "rock", "electric", "classical", "jazz", "rb"]
    if not os.path.exists('music.npy') or not os.path.exists('labels.npy'):
        for label, genre in enumerate(genre_list):
            path = os.path.join(fpath, genre)

            for fname in os.listdir(path):
                if fname[-4:] not in ('.mp3', '.MP3'):
                    continue
                if flag:
                    music = read_mp3(os.path.join(path, fname))
                    print("fname")
                    flag = False
                else:
                    music = np.vstack((music, read_mp3(os.path.join(path, fname))))

                _labels.append(label)

        labels = np.array(_labels)
        music = music.reshape(-1, 100, 38)
        np.save('music.npy', music)
        np.save('labels.npy', labels)
    else:
        #########################################################
        labels = np.load('labels.npy')
        music = np.load('music.npy')
        music = music.reshape(-1, 100, 38)

    if one_hot:
        labels = dense_to_one_hot(labels)

    assert music.shape[0] == labels.shape[0]

    perm = np.arange(music.shape[0])
    np.random.shuffle(perm)
    music = music[perm]
    labels = labels[perm]

    TRAIN_SIZE = int(0.6 * music.shape[0])
    VALIDATION_SIZE = int(0.8 * music.shape[0])

    train_music = music[:]
    train_labels = labels[:]

    validation_music = music[TRAIN_SIZE:VALIDATION_SIZE]
    validation_labels = labels[TRAIN_SIZE:VALIDATION_SIZE]

    test_music = music[VALIDATION_SIZE:]
    test_labels = labels[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_music, train_labels)
    data_sets.validation = DataSet(validation_music, validation_labels)
    data_sets.test = DataSet(test_music, test_labels)
    return data_sets
