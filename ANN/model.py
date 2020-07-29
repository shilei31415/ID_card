from data_base import image_cols, image_rows
import tensorflow as tf

# 获取卷积核
def get_weight(shape):
    W=tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.1))
    return W

# 自己定义卷积 方便后面调用
def convolutional(input, output_channels, kernal_size, stride, name, function="leaky_relu"):
    shape=input.shape.as_list()
    W=get_weight((kernal_size, kernal_size, shape[3], output_channels))
    strides=(1, stride, stride, 1)
    conv=tf.nn.conv2d(input, W, strides, padding="SAME", name=name)

    bias=tf.Variable(tf.zeros(shape=(output_channels)))
    conv = tf.nn.bias_add(conv, bias)

    if function=="leaky_relu":
        result=tf.nn.leaky_relu(conv, alpha=0.1)
        return result
    elif function=="linear":
        return conv

# 自定义池化方便调用
def avg_pool(x, stride):
    return tf.nn.avg_pool(x, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding="VALID")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="VALID")

def model(x, target = None):
    print("x", x.shape)

    conv1 = convolutional(x, 8, 5, 2, "conv1")
    pool1 = avg_pool(conv1, 4)
    print("pool1", pool1.shape)

    conv2 = convolutional(pool1, 16, 3, 1, "conv2")
    pool2 = max_pool(conv2)
    print("pool2", pool2.shape)

    conv3 = convolutional(pool2, 64, 2, 1, "conv3")
    pool3 = max_pool(conv3)
    print("pool3", pool3.shape)

    conv4 = convolutional(pool3, 128, 2, 1, "conv4")
    pool4 = max_pool(conv4)
    print("pool4", pool4.shape)

    pool_shape = pool4.get_shape().as_list()
    node = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool4, [-1, node])
    print("reshaped", reshaped.shape)

    fc1_w = get_weight([node, int(node / 2)])
    fc1_b = get_weight([int(node / 2)])
    fc1 = tf.nn.leaky_relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    print("fc1", fc1.shape)

    fc2_w = get_weight([int(node / 2), int(node / 2)])
    fc2_b = get_weight([int(node / 2)])
    fc2 = tf.nn.leaky_relu(tf.matmul(fc1, fc2_w) + fc2_b)
    print("fc2", fc2.shape)

    fc3_w = get_weight([int(node / 2), int(node / 4)])
    fc3_b = get_weight([int(node / 4)])
    fc3 = tf.nn.leaky_relu(tf.matmul(fc2, fc3_w) + fc3_b)
    print("fc3", fc3.shape)

    fc4_w = get_weight([int(node / 4), 18*12])
    fc4_b = get_weight([18*12])
    fc4 = tf.matmul(fc3, fc4_w) + fc4_b
    print("fc4", fc4.shape)

    ID = tf.sigmoid(tf.reshape(fc4, [-1, 18, 12]), "ID")
    print("ID", ID.shape)
    if target is None:
        return ID
    else:
        loss_obj = tf.reduce_mean(-(target*tf.log(ID+1e-4)+10*(1-target)*tf.log(1-ID+1e-4)))
        return ID, loss_obj



if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [4, image_rows, image_cols, 3])
    target = tf.placeholder(tf.float32, [4, 18, 12])
    model(x, target)