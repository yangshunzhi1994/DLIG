## The source code for "Dual-Latent Interactive Generation for Data-Free Knowledge Distillation".

### 1. Prepare the files

To reproduce our results, please download the pre-trained teacher models from [Google Drive](https://drive.google.com/drive/folders/13vEvJZ4I8ZjdCe0G8PzegK0nwEeYZvUm?usp=sharing) and extract them as `checkpoints/pretrained`.

Alternatively, you can train a teacher model from scratch as follows.

### 2. Reproduce our results

* To reproduce the experimental results of DLIG, please refer to the commands provided in `Tuning.py`.
* `Tuning.py` contains the configurations for different datasets and teacher-student architectures, including CIFAR-10, CIFAR-100, Tiny-ImageNet, Caltech-101, and NYUv2.
* Please select and run the command corresponding to the desired experimental setting.
* Synthesized proxy data, model checkpoints, and training logs will be saved to the output directories specified in the corresponding commands.
