import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, '../utils')
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
import tf_util


def _shape_value(dim):
    return dim.value if hasattr(dim, 'value') else dim


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, num_classes=40):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = tf.shape(point_cloud)[0]
    num_point = _shape_value(point_cloud.get_shape()[1])
    if num_point is None:
        raise ValueError('PointNet basic classification model requires a static num_point dimension.')
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    
    # MLP on global point cloud vector
    net = tf.reshape(net, tf.stack([batch_size, 1024]))
    net.set_shape([None, 1024])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001, class_weights=None):
    """ pred: B*NUM_CLASSES,
        label: B, """
    if class_weights is None:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
    else:
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        sample_weights = tf.gather(class_weights, label)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss * sample_weights)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True), num_classes=40)
        print(outputs)
