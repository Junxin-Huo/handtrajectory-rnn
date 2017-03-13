import sys
import time
import numpy as np
import tensorflow as tf
from net import inference
from loader import loadDataLabelRealtime

FRAME_COUNT = 1
BATCH_SIZE = 1

DATADIR = 'dataset_test'
NETPATH = 'data/net.ckpt'
EVAL_FREQUENCY = 10

def main(argv=None):
    print 'Loading......'
    start_time = time.time()
    begin_time = start_time

    data, label = loadDataLabelRealtime(DATADIR, shuffle=True, various=True)
    train_size = len(label)
    print 'Loaded %d datas.' % train_size

    elapsed_time = time.time() - start_time
    print('Loading datas with label elapsed %.1f s' % elapsed_time)
    print 'Building net......'
    start_time = time.time()

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, FRAME_COUNT , 2], name='data')
    keep_prob = tf.placeholder(tf.float32, name='prob')

    train_prediction, initial_state, final_state = inference(x, keep_prob, BATCH_SIZE)
    prediction = tf.nn.softmax(train_prediction)


    def eval_in_batches(_data, sess, state):
        feed_dict = {x: np.reshape(_data, [BATCH_SIZE, FRAME_COUNT, 2]),
                     keep_prob: 1.0}
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        tp, p, _final_state= sess.run([train_prediction, prediction, final_state], feed_dict=feed_dict)
        return tp, p, _final_state

    elapsed_time = time.time() - start_time
    print('Building net elapsed %.1f s' % elapsed_time)
    print 'Begin testing..., train dataset size:{0}'.format(train_size)
    start_time = time.time()

    saver = tf.train.Saver()

    elapsed_time = time.time() - start_time
    print('loading net elapsed %.1f s' % elapsed_time)
    start_time = time.time()

    ls = []
    with tf.Session() as sess:
        saver.restore(sess, NETPATH)
        tf.train.write_graph(sess.graph_def, '.', 'data/train.pb', False)
        for i in range(train_size):
            state = sess.run(initial_state)
            batch_data = np.reshape(data[i], [BATCH_SIZE, -1, 2])
            frame_length = batch_data.shape[1]
            ls_sub = []
            for j in range(frame_length):
                data_mini = batch_data[0, j]
                tp, p, state = eval_in_batches(data_mini, sess, state)
                label_prediction = np.argmax(p)
                ls_sub.append(label_prediction)
            ls.append(ls_sub)
            if i % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d, %.1f ms.' %
                      (i, 1000 * elapsed_time / EVAL_FREQUENCY))
                print 'True label: '
                print label[i]
                print 'Prediction: '
                print ls_sub
            sys.stdout.flush()


    sum = 0
    error = 0
    for i in range(len(ls)):
        ls_sub = np.asarray(ls[i], np.int)
        label_sub = np.asarray(label[i], np.int)
        sum_count = len(ls_sub)
        error_count = sum_count - np.sum(ls_sub == label_sub)
        sum += sum_count
        error += error_count
    error_rate = 100.0 * error / sum
    print('Total size: %d, Test error count: %d, error rate: %f%%' % (sum, error, error_rate))

    elapsed_time = time.time() - begin_time
    print('Total time: %.1f s' % elapsed_time)


if __name__ == '__main__':
    tf.app.run()
