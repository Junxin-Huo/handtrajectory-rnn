import tensorflow as tf
import tensorflow.contrib as con
from loader import VAR_LABEL, FRAME_COUNT

HIDDEN_SIZE = 8
NUM_LAYERS = 4
max_grad_norm = 2



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


    with tf.variable_scope("softmax"):
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        softmax_w = tf.Variable(tf.truncated_normal(shape=[HIDDEN_SIZE, VAR_LABEL + 1], stddev=0.01),
                                dtype=tf.float32,
                                name='softmax_w')
        softmax_b = tf.Variable(tf.constant(0.0, shape=[VAR_LABEL + 1]),
                                dtype=tf.float32,
                                name='softmax_b')
        logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='logits')


    with tf.variable_scope("output"):
        argmax = tf.nn.softmax(logits, name='argmax')
        state_0_c = tf.multiply(state[0].c, 1, name='state_0_c')
        state_0_h = tf.multiply(state[0].h, 1, name='state_0_h')
        state_1_c = tf.multiply(state[1].c, 1, name='state_1_c')
        state_1_h = tf.multiply(state[1].h, 1, name='state_1_h')
        state_2_c = tf.multiply(state[2].c, 1, name='state_2_c')
        state_2_h = tf.multiply(state[2].h, 1, name='state_2_h')
        state_3_c = tf.multiply(state[3].c, 1, name='state_3_c')
        state_3_h = tf.multiply(state[3].h, 1, name='state_3_h')

        # state_c = tf.Variable(tf.constant(0.0, shape=[NUM_LAYERS, HIDDEN_SIZE]),
        #                       trainable=False, name="state_c",
        #                       dtype=tf.float32)
        # for i in range(NUM_LAYERS):
        #     state_c[i] = tf.multiply(state[i].c, 1, name='state_c')

    return logits, _initial_state, state


def total_loss(logits, labels, batch_size):
    loss = con.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(labels, [-1])],
        [tf.ones([batch_size * FRAME_COUNT])])
    loss_ave = tf.reduce_sum(loss) / batch_size
    return loss_ave

def myTrain(loss, learning_rate, batch):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=batch)
    return train_op


