'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

import tensorflow as tf
from brook.tfutil import hist_summaries_traintest, scalar_summaries_traintest


def add_classification_losses(model, input_labels, l2=0):
    '''Add classification and L2 losses'''

    labels_flat = tf.cast(input_labels, tf.float32)
    logits_flat = model.logits
    model.a('labels_flat', labels_flat)
    model.a('logits_flat', logits_flat)

    with tf.name_scope('losses'):
        # model.a('binary_crossentropy', tf.keras.losses.BinaryCrossentropy(input_labels, model.logits))
        model.a('prob', model.logits)
        # model.a('sigm_crossentropy', tf.losses.sigmoid_cross_entropy(input_labels, model.logits))
        # model.a('cross_ent', tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=input_labels, name='cross_ent'))
        model.a('cross_ent', tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat, name='cross_ent'))
        model.a('loss_cross_ent', tf.reduce_mean(model.cross_ent, name='loss_cross_ent'), trackable=True)
        model.a('class_prediction', tf.argmax(model.prob, 1))

        model.a('prediction_correct', tf.equal(tf.cast(logits_flat, tf.int64), tf.cast(labels_flat, tf.int64), name='prediction_correct'))
        model.a('accuracy', tf.reduce_mean(tf.to_float(model.prediction_correct), name='accuracy'), trackable=True)
        hist_summaries_traintest(model.prob, model.cross_ent)
        # hist_summaries_traintest(model.prob)
        #scalar_summaries_traintest(loss_cross_ent, loss_spring, loss, accuracy)
        scalar_summaries_traintest(model.accuracy)

        reg_losses = model.losses
        if len(reg_losses) > 0:
            model.a('loss_reg', tf.add_n(reg_losses, name='reg_loss'), trackable=True)
            # model.a('loss', tf.add(model.loss_cross_ent, model.loss_reg, name='loss'), trackable=True)
            model.a('loss', model.loss_cross_ent, trackable=True)
        else:
            model.a('loss', model.loss_cross_ent, trackable=True)