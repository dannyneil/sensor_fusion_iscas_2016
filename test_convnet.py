import theano
import cPickle as pkl
import numpy as np
from data_utils import get_mnist
from matplotlib import pyplot as plt
from theano_layers import SpikeInputLayer, SpikeConv2DReLU, SpikeAvgPool2DReLU, \
                          SpikeFlatten, SpikeDenseLayerReLU, floatX
from spike_tester_theano import run_tester
import theano.tensor as T

if __name__ == '__main__':
    np.random.seed(42)  # for reproducibility

    # Set parameters
    weights = pkl.load(open('./nets/mnist_vis_spk_reg_norm_99.20.npy', 'rb'))

    # Load data
    X_train, Y_train, X_test, Y_test = get_mnist()

    # Set inputs
    batch_size = 100
    input_size = list(X_test.shape)
    input_size[0] = batch_size
    input_var = T.tensor4('input_var')
    input_time = T.scalar('time')

    # Load Net
    first_l = SpikeInputLayer('Conv99.20', input_var, input_size, input_time)
    second_l = SpikeConv2DReLU(first_l, floatX(weights[0]), shape=(32, 30, 30), border_mode='full')
    third_l = SpikeConv2DReLU(second_l, floatX(weights[1]), shape=(32, 28, 28))
    pool_l = SpikeAvgPool2DReLU(third_l, poolsize=(2, 2), shape=(32, 14, 14))
    flatten_l = SpikeFlatten(pool_l, shape=(batch_size, 6272) )
    first_dense = SpikeDenseLayerReLU(flatten_l, floatX(weights[2]))
    final_dense = SpikeDenseLayerReLU(first_dense, floatX(weights[3]))

    # Set constants
    t_start  = 0.000
    t_end    = 0.020
    dt       = 0.001
    max_rate = 1000.
    num_digs = 100

    # Compile
    print('Compiling...')
    output_spikes, output_time, updates = final_dense.get_output()
    get_output = theano.function([input_var, input_time], [output_spikes, output_time], updates=updates)
    print('Compiled.')

    # Run simulation
    print('Running network...')
    output = run_tester(X_test[:num_digs], Y_test[:num_digs], batch_size, t_end=t_end,
        dt=dt, max_rate=1000, proc_fn=get_output, reset_fn = final_dense.reset_recursively)

    # Report back
    print('Done.')
    print('Total output spikes: {}; average output spikes per digit: {:.2f}'.format(
        np.sum(output), np.mean(np.sum(output, axis=1)) ))
    guesses = np.argmax(output, axis=1)
    truth = np.argmax(Y_test[:output.shape[0]], axis=1)
    print('Final Accuracy: {:.2f} on {} test examples.'.format(np.mean(guesses==truth)*100.,
        output.shape[0]))

    plt.figure()
    plt.imshow(output, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.show()