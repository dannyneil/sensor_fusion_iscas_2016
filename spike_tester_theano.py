import numpy as np
import sys

def run_tester(X, Y, batch_size, t_end, dt, max_rate, proc_fn, reset_fn, verbose='full'):
    # Run network in sets of presentation for the full duration
    output = np.zeros(Y.shape).astype('int32')
    num_batches = int(np.ceil(float(X.shape[0]) / batch_size))
    for b_idx in range(num_batches):
        max_idx = np.min( ((b_idx+1)*batch_size, Y.shape[0]) )
        batch_idxs = range(b_idx*batch_size, max_idx)

        # Do the first batch for the full duration, then reset and do the next
        reset_fn()
        # Loop through all time
        for t in np.arange(dt, t_end, dt):
            batch = X[batch_idxs, :]
            rescale_fac = 1./(dt * max_rate)
            spike_snapshot = np.random.rand(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]) * rescale_fac
            inp_images = spike_snapshot <= batch
            out_spikes, t = proc_fn(inp_images.astype('float32'), float(t))
            output[batch_idxs, :] += out_spikes.astype('int32')
            if verbose=='full':
                print('.'),
                sys.stdout.flush()
        guesses = np.argmax(output, axis=1)
        truth = np.argmax(Y, axis=1)
        if verbose is not None:
            print('Batch {} of {} completed.  Accuracy: {:.2f}%.'.format(
                b_idx+1, num_batches, np.mean(guesses==truth)*100.))
    return output

