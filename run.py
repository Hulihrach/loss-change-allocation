import subprocess

train_args = ['--train_h5 data\\mnist_train.h5',
              '--test_h5 data\\mnist_val.h5',
              '--input_dim 28,28,1',
              '--arch resnet',
              '--opt rmsprop',
              '--lr 0.01',
              '--decay_schedule 10',
              '--num_epochs 20',
              '--large_batch_size 1',
              '--print_every 10',
              '--log_every 50',
              '--save_weights',
              '--output_dir temp_test\\resnet_2',
              '--train_batch_size 1',
              '--data_dirs D:\\cvut\\BP\\RoadDetector\\selim_sef-solution\\wdata\\AOI_5_Khartoum',
              '--wdata_dir D:\\cvut\\BP\\RoadDetector\\selim_sef-solution\\wdata',
              '--crop_size 128',
              '--eval_every 10']
train_arg = " ".join(train_args)
train_cmd = f"python train.py {train_arg}"

subprocess.call(train_cmd, shell=True)
