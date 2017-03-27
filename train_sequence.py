import sys
import time
import numpy as np
import tensorflow as tf
from net import inference, total_loss, myTrain
from loader import loadDataLabelSequence, FRAME_COUNT


BATCH_SIZE = 16

DATADIR = 'dataset_train'
NUM_EPOCHS = 5000
NETPATH = 'data/net.ckpt'
NETPATH_FINAL = 'data/net_final.ckpt'
PBPATH = 'data/train.pb'
EVAL_FREQUENCY = 200
KEEP_PROB = 1.0

def main(argv=None):
    with tf.Graph().as_default():
        print 'Start.'
        start_time = time.time()
        begin_time = start_time

        print 'Loading data.'
        data, label = loadDataLabelSequence(DATADIR, BATCH_SIZE)
        batch_len = label.shape[0]
        epoch_size = label.shape[1]
        train_size = batch_len * epoch_size * FRAME_COUNT
        print 'Loaded %d * %d * %d datas.' % (batch_len, epoch_size, FRAME_COUNT)

        elapsed_time = time.time() - start_time
        print('Loading images with label elapsed %.1f s' % elapsed_time)
        print 'Building net......'
        start_time = time.time()

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, FRAME_COUNT, 2], name='data')
        y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, FRAME_COUNT])
        keep_prob = tf.placeholder(tf.float32, name='prob')

        # Train model.
        train_prediction, initial_state, final_state = inference(x, keep_prob, BATCH_SIZE)

        batch = tf.Variable(0, dtype=tf.float32, trainable=False)

        learning_rate = tf.train.exponential_decay(
            0.1,  # Base learning rate.
            batch * BATCH_SIZE * FRAME_COUNT,  # Current index into the dataset.
            train_size * 100,  # Decay step.
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
            for step in range(NUM_EPOCHS * epoch_size):
                offset = step % epoch_size
                if offset == 0:
                    state = sess.run(initial_state)
                batch_data = np.reshape(data[:, offset, :, :], [BATCH_SIZE, FRAME_COUNT, 2])
                batch_labels = np.reshape(label[:, offset, :], [BATCH_SIZE, FRAME_COUNT])

                # Train RNN net.
                feed_dict = {x: batch_data,
                             y: batch_labels,
                             keep_prob: KEEP_PROB}
                for i, (c, h) in enumerate(initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
                summary, _, l, lr, predictions, state = sess.run(
                    [merged, trainer, loss, learning_rate, train_prediction, final_state],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                if (step // epoch_size > NUM_EPOCHS * 0.9) & (l < best_validation_loss):
                    print 'Previous Saving net.'
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                    best_validation_loss = l
                    saver.save(sess, NETPATH)
                if step % EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %(step, np.float32(step) / epoch_size,
                                                            1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Net loss:%.3f, learning rate: %.6f' % (l, lr))
                sys.stdout.flush()
            print 'Saving final net.'
            saver.save(sess, NETPATH_FINAL)
            train_writer.close()

        elapsed_time = time.time() - begin_time
        print('Total time: %.1f s' % elapsed_time)

if __name__ == '__main__':
    tf.app.run()