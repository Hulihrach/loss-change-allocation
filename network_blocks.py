from tf_plus import Layers, Conv2D, BatchNormalization, Activation
from tensorflow.python import keras as tfkeras
import tensorflow as tf


# LinkNet


def conv_bn_relu(num_channel, kernel_size, stride, name, padding='same', activation='relu'):
    return [
        Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               name=name + "_conv"),
        # BatchNormalization(name=name + '_bn'),
        Activation(activation)  # , name=name + '_relu'
    ]


def deconv_bn_relu(num_channels, kernel_size, name, transposed_conv, activation='relu'):
    layers = []
    if transposed_conv:
        layers.append(tf.layers.Conv2DTranspose(num_channels, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    else:
        layers.append(tf.keras.layers.UpSampling2D())
        layers.append(Conv2D(num_channels, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                             padding='same'))
    # layers.append(BatchNormalization(name=name + '_bn'))
    layers.append(Activation(activation))
    return layers


def encoder(m, n, blocks, stride, name='encoder'):
    downsample = None
    if stride != 1 or m != n:
        downsample = [
            Conv2D(n, (1, 1), strides=(stride, stride), name=name + '_conv_downsample'),
            # BatchNormalization(name=name + '_batchnorm_downsample')
        ]

    layers = [ResidualBlockLinkNet(n, stride, downsample, name=name + '/residualBlock0')]
    for i in range(1, blocks):
        layers.append(ResidualBlockLinkNet(n, stride=1, name=name + '/residualBlock{}'.format(i)))
    return layers


def decoder(n_filters, planes, name='decoder', feature_scale=2, transposed_conv=False, activation='relu'):
    layers = [layer for layer in
              conv_bn_relu(num_channel=n_filters // feature_scale, kernel_size=1, stride=1, padding='same',
                           name=name + '/c1', activation=activation)]
    for layer in deconv_bn_relu(num_channels=n_filters // feature_scale, kernel_size=3,
                                name=name + '/dc1', transposed_conv=transposed_conv, activation=activation):
        layers.append(layer)

    for layer in conv_bn_relu(num_channel=planes, kernel_size=1, stride=1, padding='same', name=name + '/c2',
                              activation=activation):
        layers.append(layer)
    return layers


def prop_layer(layer, x):
    if isinstance(layer, list):
        for l in layer:
            x = l(x)
    else:
        x = layer(x)
    return x


class LinkNetDecoder(Layers):
    def __init__(self, enc1, enc2, enc3, enc4, enc5, filters=[64, 128, 256, 512, 512], feature_scale=4,
                 skip_first=False, transposed_conv=False):
        self.enc1 = enc1
        self.enc2 = enc2
        self.enc3 = enc3
        self.enc4 = enc4
        self.enc5 = enc5
        self.skipFirst = skip_first
        super(LinkNetDecoder, self).__init__()

        self.decoder5 = self.track_layers(decoder(filters[4], filters[3], name='decoder5', feature_scale=feature_scale,
                                                  transposed_conv=transposed_conv))
        self.add1 = self.track_layer(tfkeras.layers.Add())
        self.decoder4 = self.track_layers(decoder(filters[3], filters[2], name='decoder4', feature_scale=feature_scale,
                                                  transposed_conv=transposed_conv))
        self.add2 = self.track_layer(tfkeras.layers.Add())
        self.decoder3 = self.track_layers(decoder(filters[2], filters[1], name='decoder3', feature_scale=feature_scale,
                                                  transposed_conv=transposed_conv))
        self.add3 = self.track_layer(tfkeras.layers.Add())
        self.decoder2 = self.track_layers(decoder(filters[1], filters[0], name='decoder2', feature_scale=feature_scale,
                                                  transposed_conv=transposed_conv))
        self.add4 = self.track_layer(tfkeras.layers.Add())
        self.decoder1 = self.track_layers(decoder(filters[0], filters[0], name='decoder1', feature_scale=feature_scale,
                                                  transposed_conv=transposed_conv))

        if skip_first:
            self.concat = self.track_layer(tfkeras.layers.Concatenate())
            self.conv_bn_relu_1 = []
            for layer in conv_bn_relu(32, 3, stride=1, padding='same', name='f2_skip_1'):
                self.conv_bn_relu_1.append(self.track_layer(layer))
            self.conv_bn_relu_2 = conv_bn_relu(32, 3, stride=1, padding='same', name='f2_skip_2')
        else:
            self.conv_bn_relu_3 = []
            for layer in conv_bn_relu(32, 3, stride=1, padding='same', name='f2'):
                self.conv_bn_relu_3.append(self.track_layer(layer))

    def track_layers(self, layers):
        tracked = []
        for layer in layers:
            if isinstance(layer, ResidualBlockLinkNet):
                tracked.append(layer)
            else:
                tracked.append(self.track_layer(layer))
        return tracked

    def set_conv1(self, conv1):
        self.conv1 = conv1

    def call(self, inputs):
        x = inputs

        enc1 = prop_layer(self.enc1, x)
        # for layer in self.enc1:
        #     enc1 = layer(enc1)

        enc2 = prop_layer(self.enc2, enc1)
        # for layer in self.enc2:
        #     enc2 = layer(enc2)

        enc3 = prop_layer(self.enc3, enc2)
        # for layer in self.enc3:
        #     enc3 = layer(enc3)

        enc4 = prop_layer(self.enc4, enc3)
        # for layer in self.enc4:
        #     enc4 = layer(enc4)

        enc5 = prop_layer(self.enc5, enc4)
        # for layer in self.enc5:
        #     enc5 = layer(enc5)

        dec5 = prop_layer(self.decoder5, enc5)
        # for layer in self.decoder5:
        #     dec5 = layer(dec5)

        dec5 = self.add1([dec5, enc4])

        dec4 = prop_layer(self.decoder4, dec5)
        # dec4 = dec5
        # for layer in self.decoder4:
        #     dec4 = layer(dec4)
        dec4 = self.add2([dec4, enc3])

        dec3 = prop_layer(self.decoder3, dec4)
        # dec3 = dec4
        # for layer in self.decoder3:
        #     dec3 = layer(dec3)
        dec3 = self.add3([dec3, enc2])

        dec2 = prop_layer(self.decoder2, dec3)
        # dec2 = dec3
        # for layer in self.decoder2:
        #     dec2 = layer(dec2)
        dec2 = self.add4([dec2, enc1])

        dec1 = prop_layer(self.decoder1, dec2)
        # dec1 = dec2
        # for layer in self.decoder1:
        #     dec1 = layer(dec1)

        if self.skipFirst:
            x = self.concat([self.conv1, dec1])
            x = prop_layer(self.conv_bn_relu_1, x)
            # for layer in self.conv_bn_relu_1:
            #     x = layer(x)
            x = prop_layer(self.conv_bn_relu_2, x)
            # for layer in self.conv_bn_relu_2:
            #     x = layer(x)

        else:
            x = dec1
            x = prop_layer(self.conv_bn_relu_3, x)
            # for layer in self.conv_bn_relu_3:
            #     x = layer(x)

        return x


class ResidualBlockLinkNet(Layers):

    def __init__(self, n_filters, stride=1, downsample=None, name=None):
        super(ResidualBlockLinkNet, self).__init__(name=name)

        if downsample is None:
            self.shortcut = None
        else:
            self.shortcut = self.track_layers(downsample)

        self.conv_bn_relu = self.track_layers(conv_bn_relu(n_filters, kernel_size=3, stride=stride, name=name + '/cvbnrelu'))
        self.conv1 = self.track_layer(Conv2D(n_filters, (3, 3), name=name + '_conv2', padding='same'))
        # self.bn1 = self.track_layer(BatchNormalization(name=name + '_bn'))
        self.add_layer = self.track_layer(tfkeras.layers.Add())
        self.act1 = self.track_layer(Activation('relu'))

    def track_layers(self, layers):
        return [self.track_layer(layer) for layer in layers]

    def call(self, input_tensor):
        x = input_tensor

        if self.shortcut is None:
            shortcut = input_tensor
        else:
            shortcut = x
            for layer in self.shortcut:
                shortcut = layer(shortcut)

        for layer in self.conv_bn_relu:
            x = layer(x)

        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.add_layer([x, shortcut])
        x = self.act1(x)
        return x
