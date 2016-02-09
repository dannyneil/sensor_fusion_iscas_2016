import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal import downsample

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def on_gpu():
    return theano.config.device[:3] == 'gpu'

if on_gpu():
    from theano.sandbox.cuda import dnn

class SpikeInputLayer(object):
    def __init__(self, name, inp, output_shape, time_var):
        self.name = name + '--L' + '0'
        self.inp = inp
        self.output_shape = output_shape
        self.time_var = time_var

    def reset_recursively(self):
        pass

    def get_output_shape(self):
        return self.output_shape

    def get_time_var(self):
        return self.time_var

    def get_output(self):
        return self.inp, self.time_var, []

class SpikeFlatten(object):
    def __init__(self, incoming, shape):
        self.output_shape = shape
        self.incoming = incoming
        self.name = incoming.name.split('--L')[0] + '--L' + str(int(incoming.name.split('--L')[1])+1)

    def reset_recursively(self):
        # Reset parent
        self.incoming.reset_recursively()

    def get_output_shape(self):
        return self.output_shape

    def get_output(self):
        # Recurse
        inp, time, updates = self.incoming.get_output()
        reshaped_inp = T.reshape(inp, self.output_shape)
        return reshaped_inp, time, updates

class SpikeDenseLayerReLU(object):
    """ batch_size x input_shape x output_shape """
    def __init__(self, incoming, weights, threshold=1.0, refractory=0.0):
        self.incoming = incoming
        input_shape = incoming.get_output_shape()

        self.name = incoming.name.split('--L')[0] + '--L' + str(int(incoming.name.split('--L')[1])+1)
        self.output_shape = (input_shape[0], weights.shape[1])
        self.mem = shared_zeros(self.output_shape, name=self.name+'mem')
        self.refrac_until = shared_zeros(self.output_shape, name=self.name+'refrac_until')
        self.threshold = threshold
        self.refractory = refractory
        self.W = weights

    def reset_recursively(self):
        # Reset parent
        self.incoming.reset_recursively()
        self.mem.set_value(floatX(np.zeros(self.mem.get_value().shape)))
        self.refrac_until.set_value(floatX(np.zeros(self.refrac_until.get_value().shape)))

    def get_output_shape(self):
        return self.output_shape

    def get_output(self):
        # Recurse
        inp, time, updates = self.incoming.get_output()
        # Get impulse
        impulse = T.dot(inp, self.W)
        # Destroy impulse if in refrac
        masked_imp = T.set_subtensor(impulse[(self.refrac_until>time).nonzero()], 0.)
        # Add impulse
        new_mem = self.mem + masked_imp
        # Store spiking
        output_spikes = new_mem > self.threshold
        # Reset neuron
        new_and_reset_mem = T.set_subtensor(new_mem[output_spikes.nonzero()], 0.)
        # Store refractory
        new_refractory = T.set_subtensor(self.refrac_until[output_spikes.nonzero()], time + self.refractory)

        updates.append( (self.refrac_until, new_refractory) )
        updates.append( (self.mem, new_and_reset_mem) )
        return (T.cast(output_spikes,'float32'), time, updates)

class SpikeConv2DReLU(object):
    """ batch_size x input_shape x output_shape """
    def __init__(self, incoming, weights, shape, threshold=1.0, refractory=0.0,
                 subsample=(1, 1), border_mode='valid'):
        self.incoming = incoming
        input_shape = incoming.get_output_shape()

        self.name = incoming.name.split('--L')[0] + '--L' + str(int(incoming.name.split('--L')[1])+1)
        self.output_shape = (input_shape[0], shape[0], shape[1], shape[2])
        self.mem = shared_zeros(self.output_shape, name=self.name+'mem')
        self.refrac_until = shared_zeros(self.output_shape, name=self.name+'refrac_until')
        self.threshold = threshold
        self.refractory = refractory
        self.subsample = subsample
        self.border_mode = border_mode
        self.W = weights

    def reset_recursively(self):
        # Reset parent
        self.incoming.reset_recursively()
        self.mem.set_value(floatX(np.zeros(self.mem.get_value().shape)))
        self.refrac_until.set_value(floatX(np.zeros(self.refrac_until.get_value().shape)))

    def get_output_shape(self):
        return self.output_shape

    def get_output(self):
        # RECURSE
        inp, time, updates = self.incoming.get_output()

        # CALCULATE SYNAPTIC SUMMED INPUT
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=inp,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=inp,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'
            conv_out = T.nnet.conv.conv2d(inp, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:inp.shape[2] + shift_x, shift_y:inp.shape[3] + shift_y]

        # UPDATE NEURONS
        #   Get impulse
        impulse = conv_out
        #   Destroy impulse if in refrac
        masked_imp = T.set_subtensor(impulse[(self.refrac_until>time).nonzero()], 0.)
        #   Add impulse
        new_mem = self.mem + masked_imp
        #   Store spiking
        output_spikes = new_mem > self.threshold
        #   Reset neuron
        new_and_reset_mem = T.set_subtensor(new_mem[output_spikes.nonzero()], 0.)
        #   Store refractory
        new_refractory = T.set_subtensor(self.refrac_until[output_spikes.nonzero()], time + self.refractory)

        # Store updates
        updates.append( (self.refrac_until, new_refractory) )
        updates.append( (self.mem, new_and_reset_mem) )

        # Finish
        return (T.cast(output_spikes,'float32'), time, updates)

class SpikeAvgPool2DReLU(object):
    """ batch_size x input_shape x output_shape """
    def __init__(self, incoming, shape, threshold=1.0, refractory=0.0,
                 poolsize=(2, 2), stride=None, ignore_border=True):
        self.incoming = incoming
        input_shape = incoming.get_output_shape()

        self.name = incoming.name.split('--L')[0] + '--L' + str(int(incoming.name.split('--L')[1])+1)
        self.output_shape = (input_shape[0], shape[0], shape[1], shape[2])
        self.mem = shared_zeros(self.output_shape, name=self.name+'mem')
        self.refrac_until = shared_zeros(self.output_shape, name=self.name+'refrac_until')
        self.threshold = threshold
        self.refractory = refractory
        self.poolsize = poolsize
        self.stride = stride
        self.ignore_border = ignore_border

    def reset_recursively(self):
        # Reset parent
        self.incoming.reset_recursively()
        self.mem.set_value(floatX(np.zeros(self.mem.get_value().shape)))
        self.refrac_until.set_value(floatX(np.zeros(self.refrac_until.get_value().shape)))

    def get_output_shape(self):
        return self.output_shape

    def get_output(self):
        # RECURSE
        inp, time, updates = self.incoming.get_output()

        # CALCULATE SYNAPTIC SUMMED INPUT
        impulse = downsample.max_pool_2d(inp, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border, mode='average_inc_pad')

        # UPDATE NEURONS
        #   Destroy impulse if in refrac
        masked_imp = T.set_subtensor(impulse[(self.refrac_until>time).nonzero()], 0.)
        #   Add impulse
        new_mem = self.mem + masked_imp
        #   Store spiking
        output_spikes = new_mem > self.threshold
        #   Reset neuron
        new_and_reset_mem = T.set_subtensor(new_mem[output_spikes.nonzero()], 0.)
        #   Store refractory
        new_refractory = T.set_subtensor(self.refrac_until[output_spikes.nonzero()], time + self.refractory)

        # Store updates
        updates.append( (self.refrac_until, new_refractory) )
        updates.append( (self.mem, new_and_reset_mem) )

        # Finish
        return (T.cast(output_spikes,'float32'), time, updates)