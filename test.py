import sys
import time
import numpy as np
import tensorflow as tf
from net import inference
from loader import loadDataLabel, FRAME_COUNT

BATCH_SIZE = 1

DATADIR = 'dataset_test'
NETPATH = 'data/net.ckpt'
EVAL_FREQUENCY = 10

def main(argv=None):
    print 'Loading......'
    start_time = time.time()
    begin_time = start_time

    data, label = loadDataLabel(DATADIR, shuffle=True, various=True)
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


    def eval_in_batches(_data, sess, state, _initial_state=initial_state):
        feed_dict = {x: np.reshape(_data, [BATCH_SIZE, FRAME_COUNT, 2]),
                     keep_prob: 1.0}
        for i, (c, h) in enumerate(_initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        tp, p = sess.run([train_prediction, prediction], feed_dict=feed_dict)
        return tp, p

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
        state = sess.run(initial_state)

        tf.train.write_graph(sess.graph_def, '.', 'data/train.pb', False)
        for i in range(train_size):
            batch_data = np.reshape(data[i, ...], [BATCH_SIZE, FRAME_COUNT, 2])
            tp, p = eval_in_batches(batch_data, sess, state)
            label_prediction = np.argmax(p, axis=1)
            ls.append(label_prediction)
            if i % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d, %.1f ms.' %
                      (i, 1000 * elapsed_time / EVAL_FREQUENCY))
                print 'True label: ', label[i]
                print 'Prediction: ', label_prediction
            sys.stdout.flush()


    ls = np.asarray(ls, np.int)
    error_count = train_size * 1 - np.sum(ls.T[FRAME_COUNT-1:FRAME_COUNT].T == label.T[FRAME_COUNT-1:FRAME_COUNT].T)
    error_rate = 100.0 * error_count / (train_size * 4)
    print('Total size: %d, Test error count: %d, error rate: %f%%' % (train_size * FRAME_COUNT, error_count, error_rate))

    elapsed_time = time.time() - begin_time
    print('Total time: %.1f s' % elapsed_time)


if __name__ == '__main__':
    tf.app.run()
