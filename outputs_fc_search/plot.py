import plot_util
import h5py
import numpy as np

for i in range(3):
    for j in range(2):
        filename = f'./{i}_{j}/gradients_adaptive'
        with h5py.File(filename, 'r') as f:
            print(f.keys())
            loss = list(f['trainloss'])

        print(loss)
        filename = f'./{i}_{j}/helped'
        with h5py.File(filename, 'r') as f:
            helped = np.array(f['helped'])
            plot_util.percent_helped_histograms(helped, plot_dest=f'./{i}_{j}/helped_histogram')
            plot_util.plot_approx_loss(loss, helped, plot_dest=f'./{i}_{j}/approx_loss')
            plot_util.plot_trajectory_per_layer(helped, ['FC 1', 'FC 2', 'FC 3'], [(784, 100), (100, 50), (50, 10)],
                                                plot_dest=f'./{i}_{j}/trajectories_per_layer')
