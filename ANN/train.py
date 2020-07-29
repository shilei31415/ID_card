from model2 import model
from data_base2 import image_cols, image_rows, data_base
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import graph_util

def save_pb(sess):
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['result'])
    with tf.gfile.FastGFile("savePB/model.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

batch = 8

x = tf.placeholder(tf.float32, [batch, image_rows, image_cols, 3], name = "image")
target = tf.placeholder(tf.float32, [batch, 18, 12])

ID, Loss = model(x, target)
global_step=tf.Variable(0,trainable=False)
train_step=tf.train.AdamOptimizer(1e-3).minimize(Loss,global_step=global_step)

saver = tf.train.Saver()

data = data_base("/home/shilei/CLionProjects/ID_card/")

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state("./checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for step in range(100000):
        images, labels = data.get_data(batch)
        _, loss, id= sess.run([train_step, Loss, ID],
                              feed_dict={x:images, target:labels})

        id = np.array(id)
        print(loss, np.max(id), np.min(id))
        # if step%20==0:
        #     print(id)

        if step%10==0 and loss<1:
            save_pb(sess)
            saver.save(sess, "./checkpoints/model", global_step=step)
