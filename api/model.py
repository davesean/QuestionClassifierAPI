import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
import os
import glob
import numpy as np

tf_out_path = "."


class QuestionClassifier(object):
    def __init__(self, ckp_dir=None, **kwargs):
        self.sess = tf.Session()
        self.num_nodes = 8
        self.num_features = 2
        self.max_answers = 0
        self.__dict__.update(kwargs)
        self.scaler = StandardScaler()

        self.input = tf.placeholder(tf.float32, shape=[None, self.num_features])
        self.target = tf.placeholder(tf.float32, shape=[None,])
        self.hidden = tf.contrib.layers.fully_connected(self.input, self.num_nodes,
                                                        activation_fn=tf.nn.sigmoid, scope='hidden')
        self.out = tf.contrib.layers.fully_connected(self.hidden, 1, activation_fn=None, scope='out')
        self.out_val = tf.nn.sigmoid(self.out)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(self.target, axis=-1),
                                                            logits=self.out)
        self.T_vars = tf.trainable_variables()

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.net_opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss, var_list=self.T_vars)

        self.saver = tf.train.Saver()

        if ckp_dir is not None:
            print("Checking for Checkpoint in %s" % ckp_dir)
            self.scaler = load(os.path.join(ckp_dir, "sc.bin"))
            self.load_ckp(ckp_dir)

    def fit(self, features, labels, epochs, batch_size, validation_set=(None, None)):
        scaled_features = self.scaler.fit_transform(features)
        classified_labels = np.zeros_like(labels)
        p_idx = labels > self.max_answers
        classified_labels[p_idx] = 1

        self.sess.run(tf.global_variables_initializer())

        batch_times = int(scaled_features.shape[0] / batch_size)
        input_split = np.array_split(scaled_features, batch_times)
        target_split = np.array_split(classified_labels, batch_times)

        tmp_loss = 1

        print("INFO: Beginning Training")
        for e in range(epochs):
            avg_loss = []
            for it, _ in enumerate(input_split):
                cur_loss, _ = self.sess.run([self.loss, self.net_opt], feed_dict={
                    self.input: input_split[it],
                    self.target: target_split[it]
                })
                avg_loss.append(cur_loss)
            if validation_set[0] is not None:
                loss_eval = self.predict_eval(validation_set[0], validation_set[1])
                if loss_eval <= tmp_loss:
                    tmp_loss = loss_eval
                else:
                    break
                print("Epoch: %d, mean_loss: %f, mean_val_loss: %f" % (e+1, np.mean(avg_loss), loss_eval))
            else:
                print("Epoch: %d, mean_loss: %f" % (e+1, np.mean(avg_loss)))

        save_path = self.saver.save(self.sess, os.path.join(".", "api", "model.ckpt"))
        dump(self.scaler, os.path.join(".", "api", "sc.bin"), compress=True)
        print("Model saved to %s" % save_path)

    def predict_eval(self, features, labels):
        scaled_features = self.scaler.transform(features)
        classified_labels = np.zeros_like(labels)
        p_idx = labels > self.max_answers
        classified_labels[p_idx] = 1

        loss = self.sess.run(self.loss, feed_dict={
            self.input: scaled_features,
            self.target: classified_labels
        })

        return np.mean(loss)

    def predict(self, features):
        scaled_features = self.scaler.transform(features)
        prediction = self.sess.run(self.out_val, feed_dict={
                            self.input: scaled_features
        })

        probs = 0.5 + np.abs(0.5 - prediction)
        return np.round(prediction), probs

    def load_ckp(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        self.graph = tf.get_default_graph()
        return True
