# -*- coding:utf-8 -*-
from __future__ import print_function

"""
@author: gxjun
@file: alex_net.py
@time: 17-10-10 下午5:31
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS;

tf.app.flags.DEFINE_integer('batch_size', default_value=128, docstring='批处理');
N_TEST_IMG = 5;


class AlexNet(object):
    def __init__(self):
        pass;

    def train(self, in_data, label):
        self.conv1(in_data);
        self.pool1();
        self.conv2();
        self.pool2();
        self.fc1();
        self.fc2();
        self.defc2();
        self.defc1();
        self.deconv2();
        self.deconv1();
        depred = self.deconv_1;
        # 定义损失函数.
        decode = tf.nn.sigmoid(depred, name='sigmod');
        # 交叉商
        cross_entropy = -1. * in_data * tf.log(decode) - (1. - in_data) * tf.log(1. - decode);
        loss = tf.reduce_mean(cross_entropy);

        return loss, decode;

    def conv1(self, in_data):
        with tf.name_scope('conv1') as scope:
            self.w1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 20], dtype=tf.float32, stddev=0.1),
                                  name='weight_1');
            self.b1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[20]), name='biase_1');
            conv = tf.nn.conv2d(input=in_data, filter=self.w1, strides=[1, 1, 1, 1], padding='SAME',
                                use_cudnn_on_gpu=True);
            self.conv_1 = tf.nn.bias_add(conv, self.b1, name=scope);

    def pool1(self):
        with tf.name_scope('pool1') as scope:
            self.pool_1 = tf.nn.max_pool(self.conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name=scope);

    def conv2(self):
        with tf.name_scope('conv2') as scope:
            self.w2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 20, 50], dtype=tf.float32, stddev=0.1),
                                  name='weight_2');
            self.b2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[50]), name='biase_2');
            conv = tf.nn.conv2d(input=self.pool_1, filter=self.w2, strides=[1, 1, 1, 1], padding='SAME',
                                use_cudnn_on_gpu=True);
            self.conv_2 = tf.nn.bias_add(conv, self.b2, name=scope);

    def pool2(self):
        with tf.name_scope('pool2') as scope:
            self.pool_2 = tf.nn.max_pool(self.conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name=scope);

    def fc1(self):
        with tf.name_scope('fc1') as scope:
            reshape = tf.reshape(self.pool_2, [FLAGS.batch_size, -1]);  ##转换成一维
            # dim = reshape.get_shape()[1].value;
            pool2_shape = self.pool_2.get_shape();
            dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3];
            self.w3 = tf.Variable(tf.truncated_normal(shape=[int(dim), 500], stddev=0.1, dtype=tf.float32),
                                  name='weight_3')
            self.b3 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[500]), name='biase_3')
            self.fc_1 = tf.nn.bias_add(tf.matmul(reshape, self.w3), self.b3);
            self.relu1 = tf.nn.relu(self.fc_1, name=scope);

    def fc2(self):
        with tf.name_scope('fc2') as scope:
            self.w4 = tf.Variable(tf.truncated_normal(shape=[500, 10], stddev=0.1, dtype=tf.float32), name='weight_4');
            self.b4 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[10]), name='biase_4');
            self.fc_2 = tf.nn.bias_add(tf.matmul(self.relu1, self.w4), self.b4);
            self.relu2 = tf.nn.relu(self.fc_2, name=scope);

    def defc2(self):
        with tf.name_scope('defc2') as scope:
            self.dew4 = tf.Variable(tf.truncated_normal(shape=[10, 500], stddev=0.1, dtype=tf.float32),
                                    name='deweight_4');
            self.deb4 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[500]), name='debiase_4');
            self.defc_2 = tf.nn.bias_add(tf.matmul(self.relu2, self.dew4), self.deb4);
            self.derelu2 = tf.nn.relu(self.defc_2, name=scope);

    def defc1(self):
        with tf.name_scope('defc1') as scope:
            reshape = tf.reshape(self.pool_2, [FLAGS.batch_size, -1]);  ##转换成一维
            # dim = reshape.get_shape()[1].value;
            pool2_shape = self.pool_2.get_shape();
            dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3];

            self.dew3 = tf.Variable(tf.truncated_normal(shape=[500, int(dim)], stddev=0.1, dtype=tf.float32),
                                    name='deweight_3')
            self.deb3 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[int(dim)]), name='debiase_3')
            self.defc_1 = tf.nn.bias_add(tf.matmul(self.derelu2, self.dew3), self.deb3);
            self.derelu1 = tf.nn.relu(self.defc_1, name=scope);
            fcshape = [-1, int(pool2_shape[1]), int(pool2_shape[2]), int(pool2_shape[3])];
            self.defc = tf.reshape(self.derelu1, shape=fcshape);

    def deconv2(self):
        with tf.name_scope('deconv2') as scope:
            self.dew2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 20, 50], dtype=tf.float32, stddev=0.1),
                                    name='deweight_2');
            self.deb2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[20]), name='debiase_2');
            deconv2_2 = tf.nn.conv2d_transpose(value=self.defc, filter=self.dew2,
                                               output_shape=[int(self.defc.get_shape()[0]), 14, 14, 20],
                                               strides=[1, 2, 2, 1], padding='SAME');

            self.deconv_2 = tf.nn.bias_add(deconv2_2, self.deb2, name=scope);

    def deconv1(self):
        with tf.name_scope('deconv1') as scope:
            self.dew1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 20], dtype=tf.float32, stddev=0.1),
                                    name='deweight_1');
            self.deb1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1]), name='debiase_1');
            deconv = tf.nn.conv2d_transpose(value=self.deconv_2, filter=self.dew1,
                                            output_shape=[int(self.deconv_2.get_shape()[0]), 28, 28, 1],
                                            strides=[1, 2, 2, 1], padding='SAME');
            self.deconv_1 = tf.nn.bias_add(deconv, self.deb1, name=scope);


def checkmodel(sess, ckpt_dir='model', ckpt_file='checkpoint'):
    # 创建一个saver
    saver = tf.train.Saver();
    import os
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir);
    global_step = 0
    if os.path.exists(os.path.join(ckpt_dir, ckpt_file)):

        with open(os.path.join(ckpt_dir, ckpt_file), 'r') as file:
            line = file.readline()
            ckpt = line.split('"')[1]
            global_step = int(ckpt.split('-')[1])
        ckpt_last_state = tf.train.get_checkpoint_state(ckpt_dir);
        if ckpt_last_state and ckpt_last_state.model_checkpoint_path:
            saver.restore(sess, ckpt_last_state.model_checkpoint_path);
            print('restored from checkpoint ' + ckpt);
    else:
        print('checkpoint is not exist, new saver now !')
    return sess, saver, global_step


if __name__ == '__main__':

    mnist = input_data.read_data_sets("../MNIST_files/", one_hot=True)
    # Variables
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    alex_net = AlexNet()
    loss, decoded = alex_net.train(x, y_)
    #定义优化器
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    # 正确率计算
    correct_predict = tf.equal(tf.argmax(y_, 1), tf.argmax(decoded, 1));
    accuary = tf.reduce_mean(tf.cast(correct_predict, tf.float32));
    init = tf.initialize_all_variables()
    plt.ion()  # continuously plot
    # Train
    with tf.Session() as sess:
        sess.run(init)

        sess, saver, global_step = checkmodel(sess)
        print('Begin training...')
        step = 0;
        ppxs = None;
        try:
            for i in range(global_step, 3000):
                step = i;
                xs, ys = mnist.train.next_batch(FLAGS.batch_size)
                batch_xs, batch_ys = sess.run([tf.reshape(xs, shape=[-1, 28, 28, 1]), tf.reshape(ys, shape=[-1, 10])]);
                train_step.run({x: batch_xs, y_: batch_ys})
                if i % 20 == 0:
                    train_loss = loss.eval({x: batch_xs, y_: batch_ys})
                    print('  step, loss = %6d: %6.3f  accuary = %6.3f' % (
                        i, train_loss, accuary.eval({x: batch_xs, y_: batch_ys})))
                if i % 40 == 0:
                    xs, ys = mnist.test.next_batch(FLAGS.batch_size)
                    batch_xs, batch_ys = sess.run(
                        [tf.reshape(xs, shape=[-1, 28, 28, 1]), tf.reshape(ys, shape=[-1, 10])]);
                    test_fd = {x: batch_xs, y_: batch_ys}
                    decoded_img = decoded.eval(test_fd)
                    ppxs = xs;
                    print('  step(test),step = %6d,accuary = %6.3f' % (i, accuary.eval(test_fd)));
        except Exception as e:
            print(e)
            saver.save(sess, save_path='model/alex_', global_step=step)

        saver.save(sess, save_path='model/alex_', global_step=step);
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        batch_xs, batch_ys = sess.run(
            [tf.reshape(xs, shape=[-1, 28, 28, 1]), tf.reshape(ys, shape=[-1, 10])]);
        test_fd = {x: batch_xs, y_: batch_ys}
        decoded_img = decoded.eval(test_fd)
        ppxs = xs;
        print('  step(test),step = %6d,accuary = %6.3f' % (i, accuary.eval(test_fd)));

    x_test = mnist.test.images
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(ppxs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('result.png')
    plt.show()
