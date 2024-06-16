# ACT: Adversarial Consistency Training

## Download checkpoint

We provide the checkpoint reported in our paper.

Download CIFAR10 checkpoint from [cifar10_fid_6_0.ckpt](https://1drv.ms/u/s!Aqkz6X6nVZGDjJZoqYF6N4-wBe4IZQ?e=RVasXD).

Download ImageNet 64x64 checkpoint from [imagenet_fid_10_6.ckpt](https://1drv.ms/u/s!Aqkz6X6nVZGDjJZpahGmRv_FLxCJsA?e=M5ZAk8).

Download LSUNCAT 256x256 checkpoint from [lsun256_fid_13_0.ckpt](https://1drv.ms/u/s!Aqkz6X6nVZGDke4wSYVa96wAvLqV5w?e=bZg2WR).

## Install environment

Using code below to install the package.

```bash
pip install -r requirements.txt
```

## Evaluation

### CIFAR10

You can use the following code to run evaluation on CIFAR10:

```bash
python train_cifar10.py --dataset_path your_dataset_path --resume-from checkpoint_path --mode eval
```

### ImageNet 64x64

To evaluation ImageNet 64x64, you need first download the reference batch from [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations) or [one_driver](https://1drv.ms/u/s!Aqkz6X6nVZGDjJgDTh_gfUjmHn3AEA?e=sV0SZc).

```bash
python train_imagenet64.py --dataset_path your_dataset_path --resume-from checkpoint_file --fid_path reference_batch_file --mode eval
```

## Sample

### CIFAR10

You can use the following code to run evaluation on CIFAR10:

```bash
python train_cifar10.py --dataset_path your_dataset_path --resume-from checkpoint_path --mode sample
```

### ImageNet 64x64

```bash
python train_imagenet64.py --dataset_path your_dataset_path --resume-from checkpoint_file --mode sample
```

### LSUNCAT 256x256

```bash
python lsun_cat_256.py --dataset_path your_dataset_path --resume-from checkpoint_file --mode sample
```

## Train

### CIFAR10

You can use the following code to run evaluation on CIFAR10:

```bash
python train_cifar10.py --dataset_path your_dataset_path --resume-from checkpoint_path --mode train --ckpt_path save_path --device gpu_num
```

### ImageNet 64x64

To train ImageNet 64x64, you need first download the reference batch from [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations) or [one_driver](https://1drv.ms/u/s!Aqkz6X6nVZGDjJgDTh_gfUjmHn3AEA?e=sV0SZc). (for period evaluation)

```bash
python train_imagenet64.py --dataset_path your_dataset_path --resume-from checkpoint_file --mode train --fid_path reference_batch_file --ckpt_path save_path --device gpu_num
```

### LSUN CAT 64x64

```bash
python lsun_cat_256.py --dataset_path your_dataset_path --resume-from checkpoint_file --mode train --fid_path reference_batch_file --ckpt_path save_path --device gpu_num
```

## Convert ImageNet 64x64 dataset

You need to download ImageNet 64x64 dataset with format `.npz` from [ImageNet](https://image-net.org/).

Run code below:

```bash
python convert_imagenet_lmdb.py --path_lmdb dataset_to_save --dataset_path npz_path
```

