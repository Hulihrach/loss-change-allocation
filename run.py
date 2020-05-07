import subprocess

train_args = ['--train_h5 data\\cifar10_train.h5',
              '--test_h5 data\\cifar10_val.h5',
              '--input_dim 32,32,3',
              '--arch linknet',
              '--opt adam',
              '--lr 0.0001',
              '--decay_schedule 50,150',
              '--num_epochs 200',
              '--large_batch_size 6',
              '--print_every 10',
              '--log_every 50',
              '--save_weights',
              '--output_dir temp_test\\linknet_13',
              '--train_batch_size 6',
              '--test_batch_size 2',
              '--data_dirs D:\\cvut\\BP\\RoadDetector\\selim_sef-solution\\wdata\\AOI_2_Vegas',
              '--wdata_dir D:\\cvut\\BP\\RoadDetector\\selim_sef-solution\\wdata',
              '--crop_size 256',
              '--eval_every 10',
              '--save_every 200']
train_arg = " ".join(train_args)
train_cmd = f"python train_spacenet.py {train_arg}"

subprocess.call(train_cmd, shell=True)
