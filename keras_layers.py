from keras.layers.core import Layer
import theano.tensor as T
from theano.tensor.signal import downsample
from keras import activations, initializations, regularizers, constraints
from keras.utils.theano_utils import shared_zeros, floatX, on_gpu
from keras.utils.generic_utils import make_tuple
from keras.regularizers import ActivityRegularizer, Regularizer

if on_gpu():
    from theano.sandbox.cuda import dnn

class Convolution2DNoBias(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)

        super(Convolution2DNoBias, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        #self.b = shared_zeros((nb_filter,))

        #self.params = [self.W, self.b]
        self.params = [self.W]

        self.regularizers = []

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        #self.b_regularizer = regularizers.get(b_regularizer)
        #if self.b_regularizer:
        #    self.b_regularizer.set_param(self.b)
        #    self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        #self.b_constraint = constraints.get(b_constraint)
        #self.constraints = [self.W_constraint, self.b_constraint]
        self.constraints = [self.W_constraint]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'

            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, shift_y:X.shape[3] + shift_y]

        return self.activation(conv_out)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "nb_filter": self.nb_filter,
                "stack_size": self.stack_size,
                "nb_row": self.nb_row,
                "nb_col": self.nb_col,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "border_mode": self.border_mode,
                "subsample": self.subsample,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                }

class AvgPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
        super(AvgPooling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.stride = stride
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border, mode='average_inc_pad')
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "poolsize": self.poolsize,
                "ignore_border": self.ignore_border,
                "stride": self.stride}


class DenseNoBias(Layer):
    '''
        Just your regular fully connected NN layer, without bias.
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, name=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, ):

        super(DenseNoBias, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))

        #self.params = [self.W, self.b]
        self.params = [self.W]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        #self.b_regularizer = regularizers.get(b_regularizer)
        #if self.b_regularizer:
        #    self.b_regularizer.set_param(self.b)
        #    self.regularizers.append(self.b_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        #self.b_constraint = constraints.get(b_constraint)
        #self.constraints = [self.W_constraint, self.b_constraint]
        self.constraints = [self.W_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        #self.b.name = '%s_b' % name

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W))
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}