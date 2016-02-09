import theano
import numpy as np

def normalize_net_for_spiking(model, X, active_norm=True):
    if not active_norm:
        print('NOT ACTUALLY NORMALIZING.')
    # Loop through all layers, looking for activation layers
    for idx, curr_layer in enumerate(model.layers):
        print('Beginning Layer {}: {}'.format(idx, curr_layer))
        get_layer_activ = theano.function([model.layers[0].input],
            curr_layer.get_output(train=False), allow_input_downcast=True)
        out_X = get_layer_activ(X[:1,:])
        print('Activation shape: {}'.format(out_X.shape))
        # Check for layers to normalize
        if curr_layer.__class__.__name__ == 'Activation':
            print('Calculating output...')
            out_X = get_layer_activ(X)
            max_val = np.max(np.max(out_X))
            print('Done. Max val: {}.'.format(max_val))
            if active_norm:
                model.layers[idx-1].set_weights(model.layers[idx-1].get_weights() / max_val)
                print('Modified.')
    return model