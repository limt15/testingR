import os
import sys
import math
# BASE_DIR = os.path.dirname(__file__)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tensorflow as tf
import numpy as np
import tf_util_dg
from pointnet_util_spec import pointnet_sa_module
from pointnet_util_spec import pointnet_sa_module_spec
from transform_nets import input_transform_net
# from compact_bilinear_pooling import compact_bilinear_pooling_layer


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl



def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])
    end_points['l0_xyz'] = l0_xyz  ## it in localspecgcn but not in dgcnn

    # dgcnn
    # batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    # end_points = {}
    k = 20

    # Set spectral abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_spec(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=32, mlp=[128,256], mlp2=[256], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2' , knn=True , spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_spec(l2_xyz, l2_points, npoint=32, radius=0.4, nsample=8, mlp=[256,512], mlp2=[512], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3' , knn=True, spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='hier_cluster_pool', csize = 2 )
    l4_xyz, l4_points, l4_indices = pointnet_sa_module_spec(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4', knn=True , spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max')
    bottom1 = tf.reshape(l4_points, [batch_size, -1])


    #Set spatial abstraction layers
    adj_matrix = tf_util_dg.pairwise_distance(point_cloud)
    nn_idx = tf_util_dg.knn(adj_matrix, k=k)
    edge_feature = tf_util_dg.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=6)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    adj_matrix = tf_util_dg.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util_dg.knn(adj_matrix, k=k)
    edge_feature = tf_util_dg.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
    net = tf_util_dg.conv2d(edge_feature, 64, [1, 1],padding='VALID', stride=[1, 1],bn=True, is_training=is_training,scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util_dg.pairwise_distance(net)
    nn_idx = tf_util_dg.knn(adj_matrix, k=k)
    edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util_dg.conv2d(edge_feature, 64, [1, 1],padding='VALID', stride=[1, 1],bn=True, is_training=is_training,scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util_dg.pairwise_distance(net)
    nn_idx = tf_util_dg.knn(adj_matrix, k=k)
    edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util_dg.conv2d(edge_feature, 64, [1, 1],padding='VALID', stride=[1, 1],bn=True, is_training=is_training,scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    adj_matrix = tf_util_dg.pairwise_distance(net)
    nn_idx = tf_util_dg.knn(adj_matrix, k=k)
    edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)
    net = tf_util_dg.conv2d(edge_feature, 128, [1, 1],padding='VALID', stride=[1, 1],bn=True, is_training=is_training,scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    net = tf_util_dg.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],padding='VALID', stride=[1, 1],bn=True, is_training=is_training,scope='agg', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=1, keep_dims=True)

    bottom2 = tf.reshape(net, [batch_size, -1])
    result= tf.multiply(bottom1, bottom2)



    # Fully connected layers
    net = tf.reshape(result, [batch_size, -1])
    net = tf_util_dg.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util_dg.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util_dg.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util_dg.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util_dg.fully_connected(net, 40, activation_fn=None, scope='fc3')
    return net, end_points





#################################################################################


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss



if __name__=='__main__':  # the main function of dgcnn is different with the main function of localspecgcn
    with tf.Graph().as_default():
        inputs = tf.zeros((2,512,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
