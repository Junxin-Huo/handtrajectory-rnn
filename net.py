import tensorflow as tf
import tensorflow.contrib as con
from loader import VAR_LABEL, FRAME_COUNT

HIDDEN_SIZE = 16
NUM_LAYERS = 1
max_grad_norm = 1



def inference(data, prob, BATCH_SIZE):
    with tf.variable_scope("RNN"):
        lstm_cell = con.rnn.LSTMCell(HIDDEN_SIZE)
        lstm_cell_drop = con.rnn.DropoutWrapper(lstm_cell, output_keep_prob=prob)
        cell = con.rnn.MultiRNNCell([lstm_cell_drop for _ in range(NUM_LAYERS)])
        _initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

        data = tf.nn.dropout(data, prob)
        # inputs = tf.unstack(data, num=FRAME_COUNT, axis=1)
        # outputs, state = tf.nn.dynamic_rnn(cell, data, initial_state=cell.zero_state(BATCH_SIZE, tf.float32))

        outputs = []
        state = _initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(data.shape[1]):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(data[:, time_step, :], state)
                outputs.append(cell_output)

    # with tf.variable_scope('state'):
    #     _final_state = tf.multiply(state, 1, name='final_state')

    with tf.variable_scope("softmax"):
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        softmax_w = tf.Variable(tf.truncated_normal(shape=[HIDDEN_SIZE, VAR_LABEL + 1], stddev=0.01),
                                dtype=tf.float32,
                                name='softmax_w')
        softmax_b = tf.Variable(tf.constant(0.0, shape=[VAR_LABEL + 1]),
                                dtype=tf.float32,
                                name='softmax_b')
        logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='logits')

    return logits, _initial_state, state


def total_loss(logits, labels, batch_size):
    # frame_count = int(labels.shape[1])
    a = tf.zeros([batch_size, FRAME_COUNT / 2])
    b = tf.ones([batch_size, FRAME_COUNT - FRAME_COUNT / 2])
    weights = tf.concat([a, b], 1)
    weights2 = [tf.reshape(weights, [-1])]
    loss = con.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(labels, [-1])],
        weights2)
    loss_ave = tf.reduce_sum(loss) / batch_size
    return loss_ave

def myTrain(loss, learning_rate, batch):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=batch)
    return train_op


