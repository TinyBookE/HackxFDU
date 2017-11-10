import tensorflow as tf
import os
import numpy as np


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


def get_lstm(lstm_units):
    return tf.contrib.rnn.BasicLSTMCell(lstm_units)


def train_model(datasets, epoch=10000, lstm_units=128, lr=1e-4):
    batch_size = 20
    n_batch = 6
    num_features = datasets.train.music.shape[2]
    time_steps = datasets.train.music.shape[1]
    num_classes = datasets.train.labels.shape[1]

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x-input')
        y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('layer_RNN'):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([get_lstm(lstm_units) for _ in range(3)])
        preds, final_state = tf.nn.dynamic_rnn(
            stacked_lstm,
            x,
            dtype=tf.float32
        )
        with tf.name_scope('pred'):
            pred = tf.layers.dense(preds[:, -1, :], 256, activation=tf.nn.softmax)
            variable_summaries(pred)

    with tf.name_scope('layer_network'):
        with tf.name_scope('weight1'):
            W1 = tf.Variable(tf.truncated_normal([256, 1024], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('bias1'):
            b1 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b1)
        with tf.name_scope('dropout1'):
            L1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(pred, W1) + b1), keep_prob)
            variable_summaries(L1)

        with tf.name_scope('weight2'):
            W2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name='W2')
            variable_summaries(W2)
        with tf.name_scope('bias2'):
            b2 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b2)
        with tf.name_scope('dropou2'):
            L2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L1, W2) + b2), keep_prob)
            variable_summaries(L2)

        with tf.name_scope('weight3'):
            W3 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1), name='W3')
            variable_summaries(W3)
        with tf.name_scope('bias3'):
            b3 = tf.Variable(tf.zeros([256]) + 0.1)
            variable_summaries(b3)
        with tf.name_scope('dropout3'):
            L3 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L2, W3) + b3), keep_prob)
            variable_summaries(L3)

        with tf.name_scope('weight4'):
            W4 = tf.Variable(tf.truncated_normal([256, 6], stddev=0.1), name='W4')
            variable_summaries(W4)
        with tf.name_scope('bias4'):
            b4 = tf.Variable(tf.zeros([6]) + 0.1)
            variable_summaries(b4)
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(tf.matmul(L3, W4) + b4)
            variable_summaries(prediction)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        # train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct'):
            correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    merge = tf.summary.merge_all()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if not os.path.exists('models/'):
        os.mkdir('models/')

    writer = tf.summary.FileWriter('logs/', sess.graph)

    for i in range(200):
        for step in range(n_batch):
            b_x, b_y = datasets.train.next_batch(batch_size)
            summary, _, loss_ = sess.run([merge, train_op, loss], {x: b_x, y: b_y, keep_prob: 0.7})

            accuracy_ = sess.run(accuracy, {x: datasets.train.music, y: datasets.train.labels, keep_prob: 1.0})
            print('batch_times:', step + i*6 + 1, ' train_loss:%.4f' % loss_, ' vali_accuracy:%.3f' % accuracy_)

        writer.add_summary(summary, step + i*6 + 1)
        print('save_model')
        saver.save(sess, 'models/lstm_model.ckpt')


def test(datasets, lstm_units=128):

    num_features = datasets.train.music.shape[2]
    time_steps = datasets.train.music.shape[1]
    num_classes = datasets.train.labels.shape[1]

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x-input')
        y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('layer_RNN'):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([get_lstm(lstm_units) for _ in range(3)])
        preds, final_state = tf.nn.dynamic_rnn(
            stacked_lstm,
            x,
            dtype=tf.float32
        )
        with tf.name_scope('pred'):
            pred = tf.layers.dense(preds[:, -1, :], 256, activation=tf.nn.softmax)
            variable_summaries(pred)

    with tf.name_scope('layer_network'):
        with tf.name_scope('weight1'):
            W1 = tf.Variable(tf.truncated_normal([256, 1024], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('bias1'):
            b1 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b1)
        with tf.name_scope('dropout1'):
            L1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(pred, W1) + b1), keep_prob)
            variable_summaries(L1)

        with tf.name_scope('weight2'):
            W2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name='W2')
            variable_summaries(W2)
        with tf.name_scope('bias2'):
            b2 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b2)
        with tf.name_scope('dropou2'):
            L2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L1, W2) + b2), keep_prob)
            variable_summaries(L2)

        with tf.name_scope('weight3'):
            W3 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1), name='W3')
            variable_summaries(W3)
        with tf.name_scope('bias3'):
            b3 = tf.Variable(tf.zeros([256]) + 0.1)
            variable_summaries(b3)
        with tf.name_scope('dropout3'):
            L3 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L2, W3) + b3), keep_prob)
            variable_summaries(L3)

        with tf.name_scope('weight4'):
            W4 = tf.Variable(tf.truncated_normal([256, 6], stddev=0.1), name='W4')
            variable_summaries(W4)
        with tf.name_scope('bias4'):
            b4 = tf.Variable(tf.zeros([6]) + 0.1)
            variable_summaries(b4)
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(tf.matmul(L3, W4) + b4)
            variable_summaries(prediction)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct'):
            correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    saver = tf.train.Saver()
    if not os.path.exists('models/'):
        raise 'NoModelsFound'
    saver.restore(sess, 'models/lstm_model.ckpt')

    accuracy_ = sess.run(accuracy, {x: datasets.train.music, y: datasets.train.labels, keep_prob: 1.0})
    print('test accuracy:%0.3f' % accuracy_)


def predict(seq_, lstm_units=128):
    seq = seq_.reshape(1,100,38)
    time_steps = 100
    num_features = 38
    print('init',seq.shape)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x-input')
        keep_prob = tf.placeholder(tf.float32)

    
    with tf.name_scope('layer_RNN'):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([get_lstm(lstm_units) for _ in range(3)])
        preds, final_state = tf.nn.dynamic_rnn(
            stacked_lstm,
            x,
            dtype=tf.float32
        )
        with tf.name_scope('pred'):
            pred = tf.layers.dense(preds[:, -1, :], 256, activation=tf.nn.softmax)
            variable_summaries(pred)
    
    with tf.name_scope('layer_network'):
        with tf.name_scope('weight1'):
            W1 = tf.Variable(tf.truncated_normal([256, 1024], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('bias1'):
            b1 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b1)
        with tf.name_scope('dropout1'):
            L1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(pred, W1) + b1), keep_prob)
            variable_summaries(L1)

        with tf.name_scope('weight2'):
            W2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name='W2')
            variable_summaries(W2)
        with tf.name_scope('bias2'):
            b2 = tf.Variable(tf.zeros([1024]) + 0.1)
            variable_summaries(b2)
        with tf.name_scope('dropou2'):
            L2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L1, W2) + b2), keep_prob)
            variable_summaries(L2)

        with tf.name_scope('weight3'):
            W3 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1), name='W3')
            variable_summaries(W3)
        with tf.name_scope('bias3'):
            b3 = tf.Variable(tf.zeros([256]) + 0.1)
            variable_summaries(b3)
        with tf.name_scope('dropout3'):
            L3 = tf.nn.dropout(tf.nn.tanh(tf.matmul(L2, W3) + b3), keep_prob)
            variable_summaries(L3)
    
        with tf.name_scope('weight4'):
            W4 = tf.Variable(tf.truncated_normal([256, 6], stddev=0.1), name='W4')
            variable_summaries(W4)
        with tf.name_scope('bias4'):
            b4 = tf.Variable(tf.zeros([6]) + 0.1)
            variable_summaries(b4)
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(tf.matmul(L3, W4) + b4)
            variable_summaries(prediction)

    sess = tf.Session()
    saver = tf.train.Saver()
    if not os.path.exists('models/'):
        raise 'NoModelsFound'
    print('restore model')
    saver.restore(sess, 'models/lstm_model.ckpt')

    result = sess.run(tf.argmax(prediction,1), {x: seq, keep_prob: 1.0})
    print(result)
    return result
