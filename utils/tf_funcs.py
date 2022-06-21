#!/usr/bin/env python
# encoding: utf-8
# @author: cq
# reference: zxding
# email: xinhchen2-c@my.cityu.edu.hk

import numpy as np
import tensorflow as tf
import os

def load_w2v(w2v_file, embedding_dim, debug=False):
    fp = open(w2v_file)
    words, _ = map(int, fp.readline().split())

    w2v = []
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    word_dict = dict()
    print('load word_embedding...')
    print('word: {} embedding_dim: {}'.format(words, embedding_dim))
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        word_dict[line[0]] = cnt
        w2v.append([float(v) for v in line[1:]])
    print('done!')
    w2v = np.asarray(w2v, dtype=np.float32)
    #w2v -= np.mean(w2v, axis = 0) # zero-center
    #w2v /= np.std(w2v, axis = 0)
    if debug:
        print('shape of w2v:',np.shape(w2v))
        word='the'
        print('id of \''+word+'\':',word_dict[word])
        print('vector of \''+word+'\':',w2v[word_dict[word]])
    return word_dict, w2v

def batch_index(length, batch_size, test=False):
    index = np.array(range(length))
    if not test: np.random.shuffle(index)
    for i in range(int( (length + batch_size -1) / batch_size ) ):
        ret = index[i * batch_size : (i + 1) * batch_size]
        if not test and len(ret) < batch_size : break
        yield ret

# tf functions
class Saver(object):
    def __init__(self, sess, save_dir, max_to_keep=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.sess = sess
        self.save_dir = save_dir
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=max_to_keep)

    def save(self, step):
        self.saver.save(self.sess, self.save_dir, global_step=step)

    def restore(self, idx=''):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        model_path = self.save_dir+idx if idx else ckpt.model_checkpoint_path # 'dir/-110'
        print("Reading model parameters from %s" % model_path)
        self.saver.restore(self.sess, model_path)


def get_weight_varible(name, shape):
        return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))

def tf_load_w2v(w2v_file, embedding_dim, embedding_type):
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:\n\n## embedding parameters ##')
    print('w2v_file-{}'.format(w2v_file))
    word_id_mapping, w2v = load_w2v(w2v_file, embedding_dim)
    print('embedding_type-{}\n'.format(embedding_type))
    if embedding_type == 0:  # Pretrained and Untrainable
        word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
    elif embedding_type == 1:  # Pretrained and Trainable
        word_embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
    elif embedding_type == 2:  # Random and Trainable
        word_embedding = get_weight_varible(shape=w2v.shape, name='word_embedding')
    return word_id_mapping, word_embedding

# def tf_load_w2v(w2v_file, embedding_dim, embedding_type):
#     print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:\n\n## embedding parameters ##')
#     print('w2v_file-{}'.format(w2v_file))
#     word_id_mapping, w2v = load_w2v(w2v_file, embedding_dim)
#     print('embedding_type-{}\n'.format(embedding_type))
#     if embedding_type == 0:  # Pretrained and Untrainable
#         return word_id_mapping, tf.constant(w2v, dtype=tf.float32, name='word_embedding')
#     w2v = w2v[1:]
#     if embedding_type == 1:  # Pretrained and Trainable
#         word_embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
#     else:  # Random and Trainable
#         word_embedding = get_weight_varible(shape=w2v.shape, name='word_embedding')
#     embed0 = tf.Variable(np.zeros([1, embedding_dim]), dtype=tf.float32, name="embed0", trainable=False)
#     return word_id_mapping, tf.concat((embed0, word_embedding), 0)

def getmask(length, max_len, out_shape):
    '''
    length shape:[batch_size]
    '''
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)


def biLSTM_multigpu(inputs,length,n_hidden,scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return tf.concat(outputs, 2) * mask


def LSTM_multigpu(inputs,length,n_hidden,scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return outputs * mask


def biLSTM_multigpu_last(inputs,length,n_hidden,scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    index = tf.range(0, batch_size) * max_len + tf.maximum((length - 1), 0)
    fw_last = tf.gather(tf.reshape(outputs[0], [-1, n_hidden]), index)  # batch_size * n_hidden
    index = tf.range(0, batch_size) * max_len
    bw_last = tf.gather(tf.reshape(outputs[1], [-1, n_hidden]), index)  # batch_size * n_hidden

    return tf.concat([fw_last, bw_last], 1)

def biLSTM(inputs,length,n_hidden,scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    return tf.concat(outputs, 2)


def LSTM(inputs,sequence_length,n_hidden,scope):
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        scope=scope
    )
    return outputs


def GNN(s_sen, doc_len):
    W = get_weight_varible('word_gnn_W', [4, 200, 50])
    w_src = get_weight_varible('word_gnn_w_src', [4, 50, 1])
    w_dst = get_weight_varible('word_gnn_w_dst', [4, 50, 1])
    b = get_weight_varible('word_gnn_b', [50])
    w1 = get_weight_varible('linear_w1', shape=[200, 200])
    b1 = get_weight_varible('linear_b1', shape=[200])
    dp_gnn = 0.1
    leaky_alpha = 0.2

    # batch, N, in_dim = s_sen.get_shape().as_list()
    # batch = 32
    N = 75
    in_dim = 200
    s_sen_ = tf.expand_dims(s_sen, 1)
    h = tf.matmul(s_sen_, W)

    attn_src = tf.matmul(tf.tanh(h), w_src)
    attn_dst = tf.matmul(tf.tanh(h), w_dst)
    attn = tf.tile(attn_src, (1, 1, 1, N)) + tf.transpose(tf.tile(attn_dst, (1, 1, 1, N)), (0, 1, 3, 2))
    attn = tf.nn.leaky_relu(attn, leaky_alpha)

    attn = tf.nn.softmax(attn, dim=-1)
    feat_out = tf.matmul(attn, h) + b

    feat_out = tf.reshape(tf.transpose(feat_out,(0, 2, 1, 3)), [-1, N, 200])
    feat_out = tf.nn.elu(feat_out)

    gate = tf.nn.sigmoid(tf.matmul(s_sen, w1) + b1)
    feat_out = gate * feat_out + (1 - gate) * s_sen

    feat_out = tf.nn.dropout(feat_out, keep_prob=dp_gnn)

    return feat_out


def att_avg(inputs, length):
    '''
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len = tf.shape(inputs)[1]
    inputs *= getmask(length, max_len, [-1, max_len, 1])
    inputs = tf.reduce_sum(inputs, 1, keepdims =False)
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    return inputs / length


def softmax_by_length(inputs, length):
    '''
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    '''
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims =True) + 1e-9
    return inputs / _sum


def att_var(inputs, length, w1, b1, w2):
    '''
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(inputs)[1], tf.shape(inputs)[2])
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden])


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
