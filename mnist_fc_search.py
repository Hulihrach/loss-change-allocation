import os
import subprocess

log_file = open("outputs_fc_search\\log.txt", "a")

i = 3
lr = 0.0005
j = 1
batch_size = 250
# for i, lr in enumerate([.1, .01, .001]):
#     for j, batch_size in enumerate([125, 500]):
os.mkdir(f"outputs_fc_search\\{i}_{j}")
train_args = ['--train_h5 data\\mnist_train.h5', '--test_h5 data\\mnist_val.h5', '--input_dim 28,28,1',
              '--arch fc', '--opt sgd', f'--lr {lr}', '--num_epochs 15', '--large_batch_size 11000',
              '--print_every 100','--log_every 50', '--save_weights',
              f'--output_dir outputs_fc_search\\{i}_{j}', f'--train_batch_size {batch_size}']

gradients_args = ['--train_h5 data\\mnist_train.h5', '--test_h5 data\\mnist_val.h5', '--input_dim 28,28,1',
                  '--arch fc', '--opt sgd', '--large_batch_size 11000', '--print_every 100',
                  f'--weights_h5 outputs_fc_search\\{i}_{j}\\weights',
                  f'--output_h5 outputs_fc_search\\{i}_{j}\\gradients_adaptive']

lca_args = ['--chunk_size 220', f'outputs_fc_search\\{i}_{j}']

train_arg = " ".join(train_args)
train_cmd = f"python train.py {train_arg}"
gradients_arg = " ".join(gradients_args)
gradients_cmd = f"python adaptive_calc_gradients.py {gradients_arg}"
lca_arg = " ".join(lca_args)
lca_cmd = f"python save_lca_stream.py {lca_arg}"

print(f'{i}_{j}: lr={lr}, train_batch_size={batch_size}', file=log_file)
for cmd in [train_cmd, gradients_cmd, lca_cmd]:
    print(cmd, file=log_file)
    process = subprocess.call(cmd, shell=True)

log_file.close()
