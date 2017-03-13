import sys
import time
import numpy as np
import tensorflow as tf
from net import inference, total_loss, myTrain
from loader import loadDataLabel, FRAME_COUNT


BATCH_SIZE = 32

DATADIR = 'dataset_train'
NUM_EPOCHS = 2000
NETPATH = 'data/net.ckpt'
PBPATH = 'data/train.pb'
EVAL_FREQUENCY = 50
KEEP_PROB = 1.0

def main(argv=None):
    with tf.Graph().as_default():
        print 'Start.'
        start_time = time.time()
        begin_time = start_time

        print 'Loading data.'
        data, label = loadDataLabel(DATADIR, shuffle=True, various=True)
        train_size = len(label)
        print 'Loaded %d datas.' % train_size

        elapsed_time = time.time() - start_time
        print('Loading images with label elapsed %.1f s' % elapsed_time)
        print 'Building net......'
        start_time = time.time()

        def get_input_x(x, offset=0, length=BATCH_SIZE):
            a = x[offset:(offset + length), ...]
            return np.reshape(a, [length, FRAME_COUNT, 2])

        def get_input_y(y, offset=0, length=BATCH_SIZE):
            b = y[offset:(offset + length), ...]
            return np.reshape(b, [length, FRAME_COUNT])

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, FRAME_COUNT , 2], name='data')
        y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, FRAME_COUNT])
        keep_prob = tf.placeholder(tf.float32, name='prob')

        # Train model.
        train_prediction, initial_state, final_state = inference(x, keep_prob, BATCH_SIZE)

        batch = tf.Variable(0, dtype=tf.float32, trainable=False)

        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size * 80,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learn', learning_rate)

        loss = total_loss(train_prediction, y, BATCH_SIZE)
        tf.summary.scalar('loss', loss)

        trainer = myTrain(loss, learning_rate, batch)

        elapsed_time = time.time() - start_time
        print('Building net elapsed %.1f s' % elapsed_time)
        print 'Begin training..., train dataset size:{0}'.format(train_size)
        start_time = time.time()
        best_validation_loss = 100000.0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('graph/train', sess.graph)

            # Inital the whole net.
            tf.global_variables_initializer().run()
            state = sess.run(initial_state)
            print('Initialized!')
            for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

                batch_data = get_input_x(offset=offset, x=data)
                batch_labels = get_input_y(offset=offset, y=label)

                # Train RNN net.
                feed_dict = {x: batch_data,
                             y: batch_labels,
                             keep_prob: KEEP_PROB}
                for i, (c, h) in enumerate(initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
                summary, _, l, lr, predictions = sess.run(
                    [merged, trainer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                if l < best_validation_loss:
                    print 'Saving net.'
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                    best_validation_loss = l
                    saver.save(sess, NETPATH)
                if step % EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %
                          (step, np.float32(step) * BATCH_SIZE / train_size,
                           1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                sys.stdout.flush()
            train_writer.close()

        elapsed_time = time.time() - begin_time
        print('Total time: %.1f s' % elapsed_time)

if __name__ == '__main__':
    tf.app.run()