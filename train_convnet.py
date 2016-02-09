from __future__ import print_function
import theano
import numpy as np
from data_utils import get_mnist
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras_layers import Convolution2DNoBias, AvgPooling2D, DenseNoBias

from spiking_utils import normalize_net_for_spiking
import cPickle as pkl
if __name__ == '__main__':
    np.random.seed(42)  # for reproducibility

    # Constants
    plt.ion()
    batch_size    = 256
    dense_size    = 128
    nb_epoch      = 10
    nb_kernels    = 32
    kernel_size   = 3
    pool_size     = 2
    filename_base = './nets/my_mnist'

    # Load data
    X_train, Y_train, X_test, Y_test = get_mnist()

    # Build the model
    print('Building model...')
    model = Sequential()
    #   Conv and Pooling
    model.add(Convolution2DNoBias(nb_kernels, 1, kernel_size, kernel_size, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2DNoBias(nb_kernels, nb_kernels, kernel_size, kernel_size))
    model.add(Activation('relu'))
    model.add(AvgPooling2D(poolsize=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    #   Reshape
    model.add(Flatten())
    #   Dense layers
    model.add(DenseNoBias(nb_kernels*(X_train.shape[-1]/pool_size)*(X_train.shape[-2]/pool_size),
        output_dim=dense_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(DenseNoBias(dense_size, Y_test.shape[-1]))
    model.add(Activation('softmax'))

    # Compile
    print('Compiling model...')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print('Finished compiling model.')

    # Show the data
    plt.figure(figsize=(8,8))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(X_train[i,0,:,:], interpolation='nearest', cmap=plt.get_cmap('Greys'))
    plt.show()

    # Train - around 4s per epoch on my GTX 980 Ti
    print('Training WITHOUT noise.')
    log = model.fit(X_train, Y_train, batch_size=batch_size,
        nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
        validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    final_score = score[1]*100.

    # Save weights
    filename_base = '{}_{:2.2f}'.format(filename_base, final_score)
    wt_filename = filename_base+'_wts.npy'
    model.save_weights(wt_filename)
    print('Saved {}.'.format(wt_filename))

    # Plot
    history = log.history
    plt.figure(figsize=(14,10))
    plt.subplot(2,1,1)
    plt.plot(history['acc'], '.-')
    plt.plot(history['val_acc'], '.-')
    plt.legend(['Accuracy', 'Validation Accuracy'],loc='lower right')
    plt.grid(which='both')
    plt.subplot(2,1,2)
    plt.semilogy(history['loss'], '.-')
    plt.semilogy(history['val_loss'], '.-')
    plt.legend(['Loss', 'Validation Loss'])
    plt.grid(which='both')
    plt.show()

    # Save feature vector
    target_layer = -3 # third from end; softmax, 128x10, dropout
    get_fv = theano.function([model.layers[0].input], model.layers[target_layer].get_output(train=False), allow_input_downcast=True)
    batch_size = 500
    #   Train FV
    train_fv_data = np.zeros([60000, dense_size])
    for b_idx in range(X_train.shape[0]/batch_size):
        curr_fv_data = get_fv(X_train[b_idx*batch_size:(b_idx+1)*batch_size, :, :, :])
        train_fv_data[b_idx*batch_size:(b_idx+1)*batch_size, :] = curr_fv_data
    print('Training Feature Vector size: {}'.format(train_fv_data.shape))
    #   Test FV
    test_fv_data = np.zeros([10000, dense_size])
    for b_idx in range(X_test.shape[0]/batch_size):
        curr_fv_data = get_fv(X_test[b_idx*batch_size:(b_idx+1)*batch_size, :, :, :])
        test_fv_data[b_idx*batch_size:(b_idx+1)*batch_size, :] = curr_fv_data
    print('Test Feature Vector size: {}'.format(test_fv_data.shape))
    #   Save
    fv_filename = filename_base+'_fvs.npy'
    pkl.dump({'train_fv': train_fv_data, 'test_fv': test_fv_data,
        'Y_train': Y_train, 'Y_test': Y_test}, open(fv_filename,'wb'))
    print('FVs saved to: {}.'.format(fv_filename))

    # Normalize the network
    model = normalize_net_for_spiking(model, X_train[:10000,:])

    # Ensure it didn't affect accuracy
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
    print('After norm, test score:', score[0])
    print('After norm, test accuracy: {:2.2f}%'.format(score[1]*100.))

    # Write out weights
    wt_norm_filename = filename_base+'_wts_normd.npy'
    model.save_weights(wt_norm_filename)
    print('Saved {}.'.format(wt_norm_filename))