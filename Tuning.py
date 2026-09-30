import os

print ('The specific model being run is:                CIFAR10,  DFKD ')
os.system("python datafree_kd.py --dataset cifar10 --adv 1.33 --bn 10 --oh 0.5 --div 1.0 --teacher resnet34 --student resnet18 --save_dir checkpoints1/c10r34r18-ours --log_tag 'c10r34r18-ours-ep320' ")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

print ('The specific model being run is:                CIFAR10,  DFKD ')
os.system("python datafree_kd.py --dataset cifar10 --adv 1.33 --bn 10 --oh 0.5 --div 1.0 --teacher wrn40_2 --student wrn16_1 --save_dir checkpoints1/c10w402w161-ours --log_tag 'c10w402w161-ours-ep320' ")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

print ('The specific model being run is:                CIFAR10,  DFKD ')
os.system("python datafree_kd.py --dataset cifar10 --adv 1.33 --bn 10 --oh 0.5 --div 1.0 --teacher wrn40_2 --student wrn16_2 --save_dir checkpoints1/c10w402w162-ours --log_tag 'c10w402w162-ours-ep320' ")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

print ('The specific model being run is:                CIFAR10,  DFKD ')
os.system("python datafree_kd.py --dataset cifar10 --adv 1.33 --bn 10 --oh 0.5 --div 1.0 --teacher wrn40_2 --student wrn40_1 --save_dir checkpoints1/c10w402w401-ours --log_tag 'c10w402w401-ours-ep320' ")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

print ('The specific model being run is:                CIFAR10,  DFKD ')
os.system("python datafree_kd.py --dataset cifar10 --adv 1.33 --bn 10 --oh 0.5 --div 1.0 --teacher vgg11 --student resnet18 --save_dir checkpoints1/c10vgg11r18-ours --log_tag 'c10vgg11r18-ours-ep320' ")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')








# print ('The specific model being run is:                cifar100,  DFKD ')
# os.system("python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 320 --dataset cifar100 --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/c100r34r18-ours --adv 1.33 --bn 10.0 --oh 0.5 --div 1.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 --log_tag c100r34r18-ours-ep320 ")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:                cifar100,  DFKD ')
# os.system("python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 320 --dataset cifar100 --lr_g 4e-3 --teacher wrn40_2 --student wrn16_1 --save_dir run/c100w402w161-ours --adv 1.33 --bn 10.0 --oh 0.5 --div 1.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 --log_tag c100w402w161-ours-ep320 ")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:                cifar100,  DFKD ')
# os.system("python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 320 --dataset cifar100 --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c100w402w162-ours --adv 1.33 --bn 10.0 --oh 0.5 --div 1.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 --log_tag c100w402w162-ours-ep320 ")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:                cifar100,  DFKD ')
# os.system("python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 320 --dataset cifar100 --lr_g 4e-3 --teacher wrn40_2 --student wrn40_1 --save_dir run/c100w402w401-ours --adv 1.33 --bn 10.0 --oh 0.5 --div 1.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 --log_tag c100w402w401-ours-ep320 ")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')

# print ('The specific model being run is:                cifar100,  DFKD ')
# os.system("python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 320 --dataset cifar100 --lr_g 4e-3 --teacher vgg11 --student resnet18 --save_dir run/c100vgg11r18-ours --adv 1.33 --bn 10.0 --oh 0.5 --div 1.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 --log_tag c100vgg11r18-ours-ep320 ")
# print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')




print ('The specific model being run is:                nyuv2, stable hierarchical DFAD ')
os.system("python -u DFAD_nyu_deeplab.py --data_root ./data/NYUv2 --ckpt checkpoints1/pretrained/nyuv2-deeplabv3_resnet50.pt --epochs 320 --epoch_itrs 50 --batch_size 64 --test_batch_size 9 --workers 0 --img_size 128 --test_size 256 --lr_S 0.1 --lr_G 0.001 --adv 1.33 --bn 2.0 --oh 0.0 --div 0.1 --confidence 0.1 --balance 0.1 --scheduler --step_size 100 --lr_gamma 0.3 --weight_decay 0.00005 --save_img --save_dir checkpoints1/dfad-nyuv2-stable2 --log_tag dfad-nyuv2-stable6-ep320")
print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')




print('The specific model being run is:                Caltech101, DFKD, B-lrS0.05', flush=True)
os.system("python -u DFAD_caltech101.py --data_root ./data/caltech101 --ckpt checkpoints1/pretrained/caltech101-resnet34.pt --epochs 320 --adv 1.33 --bn 0.8 --oh 0.1 --balance 0.1 --div 0.1 --epoch_itrs 50 --batch_size 64 --test_batch_size 32 --workers 0 --img_size 128 --lr_S 0.05 --lr_G 0.001 --scheduler --step_size 100 --lr_gamma 0.1 --weight_decay 0.0005 --save_dir checkpoints1/dfad-caltech101 --log_tag dfad-caltech101-B-lrS005")
print('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n')
