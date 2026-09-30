## The source code for "Dual-Latent Interactive Generation for Data-Free Knowledge Distillation".

### 1. Environment Setup

This implementation is based on the NAYER framework. For environment installation and dependency configuration, please refer to the official NAYER repository:

[NAYER: Noisy Layer Data Generation for Efficient and Effective Data-Free Knowledge Distillation](https://github.com/tmtuan1307/NAYER)

### 2. Prepare the files

All datasets used in our experiments are publicly available, including CIFAR-10, CIFAR-100, Tiny-ImageNet, Caltech-101, and NYUv2.

To reproduce our results, please download the pre-trained teacher models from [Google Drive](https://drive.google.com/drive/folders/13vEvJZ4I8ZjdCe0G8PzegK0nwEeYZvUm?usp=sharing) and extract them as `checkpoints/pretrained`.

Alternatively, you can train a teacher model from scratch as follows:

```bash
python train_scratch.py --model wrn40_2 --dataset cifar10 --batch-size 256 --lr 0.1 --epoch 200 --gpu 0

### 3. Reproduce our results

* To reproduce the experimental results of DLIG, please refer to the commands provided in `Tuning.py`.
* `Tuning.py` contains the configurations for different datasets and teacher-student architectures, including CIFAR-10, CIFAR-100, Tiny-ImageNet, Caltech-101, and NYUv2.
* Please select and run the command corresponding to the desired experimental setting.
* Synthesized proxy data, model checkpoints, and training logs will be saved to the output directories specified in the corresponding commands.
