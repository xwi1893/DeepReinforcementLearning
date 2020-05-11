import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, sess, n_actions, n_features, lr=0.01):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [1, self.n_features], name="state")
        self.a = tf.placeholder(tf.int32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        with tf.name_scope("Actor"):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name="l1")

            self.act_prob = tf.layers.dense(inputs=l1, units=self.n_actions, activation=tf.nn.softmax,
                                            kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                            bias_initializer=tf.constant_initializer(0.1),
                                            name="acts_prob")

        with tf.name_scope("exp_v"):
            self.loss = tf.reduce_mean(-self.td_error*tf.log(self.act_prob[0, self.a]))

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.sess.run(self.act_prob, feed_dict={self.s: s})
        return np.random.choice(self.n_actions, p=prob.ravel())

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.s: s, self.a: a, self.td_error: td})
        return loss


class Critic:
    def __init__(self, sess, n_features, lr=0.01, reward_decay=0.95):
        self.sess = sess
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay

        self.s = tf.placeholder(tf.float32, [1, self.n_features], name="state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], name="pred_v")
        self.r = tf.placeholder(tf.float32, None, name="reward")

        with tf.name_scope("Critic"):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name="l1")

            self.v = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.softmax,
                                     kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     name="l1")

        with tf.name_scope("td_error"):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.reduce_mean(-self.td_error * self.v)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error