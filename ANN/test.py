from model2 import model
from data_base import image_cols, image_rows, data_base
import tensorflow as tf
import os
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

ID_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X", "e"]

batch = 1

x = tf.placeholder(tf.float32, [batch, image_rows, image_cols, 3])

ID = model(x)
saver = tf.train.Saver()

data = data_base("/media/shilei/ʩ/picture")

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state("./checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for step in range(1000):
        images = []
        image = data.test_image()
        images.append(image)
        images = np.array(images)
        images = images.astype(np.float)
        images = images/255.0
        id = sess.run([ID], feed_dict={x:images})
        id = np.array(id)
        print(np.max(id), np.min(id))

        IDs = ID_num[np.argmax(id[0, 0, 0])]
        for i in range(1, 18):
            IDs += ID_num[np.argmax(id[0, 0, i])]
        cv2.imwrite("./result/"+IDs+".png", image)
        cv2.waitKey(1)
