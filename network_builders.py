'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import print_function
from __future__ import division

from tf_plus import BatchNormalization, Lambda, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, he_normal, relu, softmax, Activation
from tf_plus import Layers, SequentialNetwork, l2reg, PreprocessingLayers

# use tensorflow's version of keras, or else get version incompatibility errors
from tensorflow.python import keras as tfkeras
import tensorflow as tf
from network_blocks import *

'''
Methods to set up network architectures
'''

def build_lenet_conv(args): # ok this is a slightly modified lenet
    return SequentialNetwork([
        Conv2D(20, 5, kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        # BatchNormalization(momentum=0.0, name='batch_norm_1'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(40, 5, kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        # BatchNormalization(momentum=0.0, name='batch_norm_2'),
        Activation('relu'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dropout(0.25),
        Dense(400, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dropout(0.5),
        Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_2')
    ])

def build_network_fc(args):
    return SequentialNetwork([
        Flatten(),
        Dense(100, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dense(50, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
        # can also try kernel_initializer=tfkeras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
    ])

# for experimental runs
def build_network_fc_special(args):
    return SequentialNetwork([
        Flatten(),
        Dense(100, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        BatchNormalization(momentum=0, name='batch_norm_1'),
        Activation('relu'),
        Dense(50, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        BatchNormalization(momentum=0, name='batch_norm_1'),
        Activation('relu'),
        Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])

# Layer sizes are geometrically interpolated between 3072 and 10
def build_fc_adjustable(args):
    if args.num_layers == 3:
        return SequentialNetwork([
            Flatten(),
            Dense(455, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
            Dense(67, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
            Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
        ])
    elif args.num_layers == 4:
        return SequentialNetwork([
            Flatten(),
            Dense(734, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
            Dense(175, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
            Dense(42, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_3'),
            Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_4')
        ])
    elif args.num_layers == 5:
        return SequentialNetwork([
            Flatten(),
            Dense(977, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
            Dense(311, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
            Dense(99, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_3'),
            Dense(31, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_4'),
            Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_5')
        ])

# from https://arxiv.org/pdf/1412.6806.pdf
# and https://github.com/PAN001/All-CNN
def build_all_cnn(args):
    return SequentialNetwork([
        #Dropout(0.1),
        Conv2D(96, (3, 3), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        BatchNormalization(momentum=0.0, name='batch_norm_1'),
        Activation('relu'),
        Conv2D(96, (3, 3), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        BatchNormalization(momentum=0.0, name='batch_norm_2'),
        Activation('relu'),
        Conv2D(96, (3, 3), kernel_initializer=he_normal, padding='same', strides=(2, 2), name='conv2D_strided_1'),
        Dropout(0.5),
        Conv2D(192, (3, 3), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        BatchNormalization(momentum=0.0, name='batch_norm_3'),
        Activation('relu'),
        Conv2D(192, (3, 3), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        BatchNormalization(momentum=0.0, name='batch_norm_4'),
        Activation('relu'),
        Conv2D(192, (3, 3), kernel_initializer=he_normal, padding='same', strides=(2, 2), name='conv2D_strided_2'),
        Dropout(0.5),
        Conv2D(192, (3, 3), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_5'),
        BatchNormalization(momentum=0.0, name='batch_norm_5'),
        Activation('relu'),
        Conv2D(192, (1, 1), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_6'),
        BatchNormalization(momentum=0.0, name='batch_norm_6'),
        Activation('relu'),
        Conv2D(10, (1, 1), kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_7'),
        GlobalAveragePooling2D()
    ])

# small vgg-like network
def build_vgg_mini(args):
    return SequentialNetwork([
        Conv2D(64, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(256, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dense(512, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dropout(0.5),
        Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_2')
    ])

# VGG-11 (https://arxiv.org/pdf/1409.1556.pdf) but with half the channels,
# last chunk removed (2 fewer layers), and smaller FC layers for smaller images
def build_vgg_half(args):
    return SequentialNetwork([
        Conv2D(32, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(64, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_2'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_3'),
        Conv2D(128, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_4'),
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(256, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_5'),
        Conv2D(256, (3, 3), kernel_initializer=he_normal, padding='same', activation=relu, kernel_regularizer=l2reg(args.l2), name='conv2D_6'),
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dense(512, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_1'),
        Dropout(0.5),
        Dense(512, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(args.l2), name='fc_2'),
        Dropout(0.5),
        Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2), name='fc_3')
    ])

# ResNet-20
def build_resnet(args):
    return SequentialNetwork([
        # pre-blocks
        Conv2D(16, 3, kernel_initializer=he_normal, padding='same', kernel_regularizer=l2reg(args.l2), name='conv2D_1'),
        BatchNormalization(momentum=0.0, name='batch_norm_1'),
        Activation('relu'),
        # set 1
        ResidualBlock(3, 16, first_stride=(1, 1), name_prefix='1A_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 16, first_stride=(1, 1), name_prefix='1B_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 16, first_stride=(1, 1), name_prefix='1C_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),

        # set 2
        ResidualBlock(3, 32, first_stride=(2, 2), name_prefix='2A_', identity=False, resize=args.resize_less, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 32, first_stride=(1, 1), name_prefix='2B_', identity=True, resize=args.resize_less, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 32, first_stride=(1, 1), name_prefix='2C_', identity=True, resize=args.resize_less, l2=args.l2, l2_shortcut=args.l2),

        tf.layers.Conv2DTranspose(16, 2, strides=(2, 2)),

        # tf.layers.Conv2DTranspose(1, 15, padding='valid'),

        # set 3
        ResidualBlock(3, 64, first_stride=(2, 2), name_prefix='3A_', identity=False, resize=args.resize_less, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 64, first_stride=(1, 1), name_prefix='3B_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 64, first_stride=(1, 1), name_prefix='3C_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),
        # post-blocks
        # GlobalAveragePooling2D(),

        tf.layers.Conv2DTranspose(16, 1, strides=(2, 2)),

        ResidualBlock(3, 64, first_stride=(2, 2), name_prefix='4A_', identity=False, resize=args.resize_less, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 64, first_stride=(1, 1), name_prefix='4B_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),
        ResidualBlock(3, 64, first_stride=(1, 1), name_prefix='4C_', identity=True, resize=args.resize_more, l2=args.l2, l2_shortcut=args.l2),

        # tf.layers.Conv2DTranspose(32, 1, strides=(2, 2)),
        tf.layers.Conv2DTranspose(16, 1, strides=(2, 2)),
        tf.layers.Conv2DTranspose(8, 1, strides=(1, 1)),
        tf.layers.Conv2DTranspose(1, 1, strides=(1, 1)),

        # tf.layers.Conv2DTranspose(1, 5, padding='valid'),
        # tf.layers.Conv2DTranspose(1, 11, padding='valid'),
        # tf.layers.Conv2DTranspose(1, 15, padding='valid'),
        # tf.layers.Conv2DTranspose(1, 21, padding='valid'),
        # tf.layers.Conv2DTranspose(1, 42, padding='valid', name='probs'),

        Activation('sigmoid', name='mask')
        # Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(args.l2_special), name='fc_last')
    ])

def build_linknet():
    return LinkNet()

# blocks used as part of a SequentialNetwork
class ResidualBlock(Layers):
    # based on https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

    def __init__(self, kernel_size, filters, first_stride, name_prefix='', identity=True, resize=1, l2=0, l2_shortcut=0, *args, **kwargs):
        super(ResidualBlock, self).__init__(*args, **kwargs)
        self.identity = identity
        self.conv1 = self.track_layer(Conv2D(int(filters * resize), kernel_size, strides=first_stride, kernel_initializer=he_normal,
            padding='same', kernel_regularizer=l2reg(l2), name=name_prefix+'conv2D_1'))
        self.bn1 = self.track_layer(BatchNormalization(momentum=0.0, name=name_prefix+'batch_norm_1'))
        self.act1 = self.track_layer(Activation('relu'))

        self.conv2 = self.track_layer(Conv2D(filters, kernel_size, strides=(1, 1), kernel_initializer=he_normal,
            padding='same', kernel_regularizer=l2reg(l2), name=name_prefix+'conv2D_2'))
        self.bn2 = self.track_layer(BatchNormalization(momentum=0.0, name=name_prefix+'batch_norm_2'))

        if not self.identity:
            self.conv_shortcut = self.track_layer(Conv2D(filters, (1, 1), strides=first_stride, kernel_initializer=he_normal,
                padding='same', kernel_regularizer=l2reg(l2_shortcut), name=name_prefix+'shortcut_conv'))

        self.add_layer = self.track_layer(tfkeras.layers.Add())
        self.act2 = self.track_layer(Activation('relu')) # TODO need relu on last block?

    def call(self, input_tensor):
        first_layer = self.act1(self.bn1(self.conv1(input_tensor)))
        second_layer = self.bn2(self.conv2(first_layer))
        if self.identity:
            reshaped_input = input_tensor
        else:
            reshaped_input = self.conv_shortcut(input_tensor)
        return self.act2(self.add_layer([reshaped_input, second_layer]))


class LinkNet(Layers):

    def __init__(self, classes=1, dropout=0.5, feature_scale=4):
        super(LinkNet, self).__init__()

        self.conv_bn_relu_1 = []
        for layer in conv_bn_relu(32, 3, stride=1, name="block1_conv1"):
            self.conv_bn_relu_1.append(self.track_layer(layer))

        self.conv_bn_relu_2 = []
        for layer in conv_bn_relu(32, 3, stride=1, name="block1_conv2"):
            self.conv_bn_relu_2.append(self.track_layer(layer))
        self.maxPool = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="block1_pool")

        layers = [1, 1, 1, 1, 1]
        # filters = [64, 128, 256, 512, 512]
        filters = [64, 128, 256, 512, 32]
        enc1 = self.track_layers(encoder(m=32, n=filters[0], blocks=layers[0], stride=1, name='encoder1'))
        enc2 = self.track_layers(encoder(m=filters[0], n=filters[1], blocks=layers[1], stride=2, name='encoder2'))
        enc3 = self.track_layers(encoder(m=filters[1], n=filters[2], blocks=layers[2], stride=2, name='encoder3'))
        enc4 = self.track_layers(encoder(m=filters[2], n=filters[3], blocks=layers[3], stride=2, name='encoder4'))
        enc5 = self.track_layers(encoder(m=filters[3], n=filters[4], blocks=layers[4], stride=2, name='encoder5'))

        self.decoder = self.track_layer(LinkNetDecoder(enc1, enc2, enc3, enc4, enc5, filters, feature_scale))
        # self.dropout = tfkeras.layers.SpatialDropout2D(dropout)
        self.conv1 = self.track_layer(Conv2D(filters=classes, kernel_size=(1, 1), padding='same', name='prediction'))
        self.act = self.track_layer(Activation('relu', name='mask'))
        self.flat = self.track_layer(Flatten())
        self.fc1 = self.track_layer(Dense(400, kernel_initializer=he_normal, activation=relu, kernel_regularizer=l2reg(0), name='fc_1'))
        # Dropout(0.5),
        self.fc2 = self.track_layer(Dense(10, kernel_initializer=he_normal, activation=None, kernel_regularizer=l2reg(0), name='fc_2'))

    def track_layers(self, layers):
        return [self.track_layer(layer) for layer in layers]

    def call(self, inputs):
        x = inputs
        for layer in self.conv_bn_relu_1:
            x = layer(x)

        conv1 = x
        self.decoder.set_conv1(conv1)

        for layer in self.conv_bn_relu_2:
            x = layer(x)

        x = self.maxPool(x)
        x = self.decoder(x)
        # x = self.dropout(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
