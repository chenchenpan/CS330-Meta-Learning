import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                    filters=num_filter,
                    kernel_size=3,
                    padding='SAME',
                    activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out

def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    """
        calculates the prototype network loss using the latent representation of x
        and the latent representation of the query set
        Args:
            x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
            q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
            labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
            num_classes: number of classes (N) for classification
            num_support: number of examples (S) in the support set
            num_queries: number of examples (Q) in the query set
        Returns:
            ce_loss: the cross entropy loss between the predicted labels and true labels
            acc: the accuracy of classification on the queries
    """
    #############################
    #### YOUR CODE GOES HERE ####

    # compute the prototypes
    x_latent = tf.reshape(x_latent, [num_classes, num_support, -1])
    ck = tf.reduce_mean(x_latent, 1)
    # print('ck' * 20)
    # print(ck)
    # print('=' * 25)

    # compute the distance from the prototypes
    q_latent = tf.reshape(q_latent, [num_classes*num_queries, 1, -1])
    q_latent = tf.tile(q_latent, [1, num_classes, 1])
    ck = tf.reshape(ck, [1, num_classes, -1])
    ck_tile = tf.tile(ck, [num_classes*num_queries, 1, 1])
    # distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(q_latent, ck_tile)), axis=-1))
    distance = tf.reduce_sum(tf.square(tf.subtract(q_latent, ck_tile)), axis=-1)
    # distance = tf.reduce_euclidean_norm((q_latent - ck_tile), -1)
    logits = tf.reshape(-distance, [num_classes, num_queries, num_classes])

    # compute cross entropy loss
    ce_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits))
    correct = tf.equal(tf.argmax(logits, -1), tf.argmax(labels_onehot, -1))
    correct = tf.cast(correct, tf.float32)
    acc = tf.reduce_mean(correct)
    # note - additional steps are needed!

    # return the cross-entropy loss and accuracy
    # ce_loss, acc = None, None
    #############################
    return ce_loss, acc

