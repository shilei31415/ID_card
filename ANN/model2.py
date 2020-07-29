from data_base import image_cols, image_rows
import tensorflow as tf

# 获取卷积核
def get_weight(shape):
    W=tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.1))
    return W

# 自己定义卷积 方便后面调用
def convolutional(input, output_channels, kernal_size, stride, name, padding="SAME", function="leaky_relu"):
    shape=input.shape.as_list()
    W=get_weight((kernal_size, kernal_size, shape[3], output_channels))
    strides=(1, stride, stride, 1)
    conv=tf.nn.conv2d(input, W, strides, padding=padding, name=name)

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

def max_pool(x, stride):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding="VALID")

def model(x, target = None):
    print(x.shape)
    # (4, 40, 448, 3)

    conv1 = convolutional(x, 64, 5, 1, "conv1")
    pool1 = avg_pool(conv1, 2)
    print(pool1.shape)
    # (4, 20, 224, 64)

    conv2 = convolutional(pool1, 128, 3, 1, "conv2")
    pool2 = max_pool(conv2, 2)
    print(pool2.shape)
    # (4, 10, 112, 128)

    conv3 = convolutional(pool2, 64, 3, 1, "conv3")
    pool3 = max_pool(conv3, 2)
    print(pool3.shape)
    # (4, 5, 56, 64)

    conv4 = convolutional(pool3, 32, 3, 1, "conv4", "VALID")
    print(conv4.shape)
    # (4, 3, 54, 32)

    conv5 = convolutional(conv4, 12, 3, 3, "conv5", "VALID")
    print(conv5.shape)
    # (4, 1, 18, 12)

    ID = tf.sigmoid(tf.reshape(conv5, [-1, 18, 12]), "result")
    print("ID", ID.shape)
    # ID (4, 18, 12)
    if target is None:
        return ID
    else:
        loss_obj = tf.reduce_mean(-(target*tf.log(ID+1e-4)+10*(1-target)*tf.log(1-ID+1e-4)))
        return ID, loss_obj



if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [4, 40, 448, 3])
    target = tf.placeholder(tf.float32, [4, 18, 12])
    model(x, target)