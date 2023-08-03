## learning useful representations for shifting tasks and distributions
-----------------------
Official Pytorch implementation of [paper](https://arxiv.org/abs/2212.07346)

[Jianyu Zhang](https://www.jianyuzhang.com/),  [Léon Bottou](https://leon.bottou.org/)


## requirements

- python==3.7
- torch>=1.13.1  
- torchvision>=0.14.1
- pyyaml==6.0
- classy-vision==0.6.0

## datasets

We consider the following datasets: 
- [ImageNet](https://www.image-net.org/index.php) 
- [Inaturalist18](https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz)
- [Cifar10/Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

Download and extract `ImageNet` and `Inaturalist18` datasets to `data/imagenet` and `data/inaturalist18`. The resulting folder structure should be:

```
📦 RRL
 ┣ 📂data
 ┃ ┣ 📂imagenet
 ┃ ┣ 📂inaturalist18
```


## Supervised transfer learning (ResNet)


### Download (ImageNet1k) pretrained checkpoints:

You can get pretrained checkpoints either:
- by automatically download according to ```python tools download.py``` or
- by manually download according to [download_checkpoint.md](download_checkpoint.md) or 
- by training from scratch according to [download_checkpoint.md](download_checkpoint.md)


The resulting folder structure should be: 
```
📦 RRL
 ┣ 📂checkpoints
 ┃ ┣ 📂supervised_pretrain
 ┃ ┃ ┣ 📂resnet50
 ┃ ┃ ┃ ┣📜 checkpoint_run0.pth.tar 
 ┃ ┃ ┃ ┃ ...            
 ┃ ┃ ┃ ┗📜 checkpoint_run9.pth.tar 
 ┃ ┃ ┣📜 2resnet50_imagenet1k_supervised.pth.tar
 ┃ ┃ ┣📜 4resnet50_imagenet1k_supervised.pth.tar
 ┃ ┃ ┣📜 resnet50w2_imagenet1k_supervised.pth.tar
 ┃ ┃ ┗📜 resnet50w4_imagenet1k_supervised.pth.tar
 ┃ ┃ ┗📜 resnet50_imagenet1k_supervised_distill5.pth.tar

```


### Transfer by Linear Probing, Fine-Tuning, and Two-stage Fine-Tuning:


|method|architecture| target task |linear probing| fine-tuning | two-stage fine-tuning |
|:---:|:---:       |:---:        |:---:       |:---:       |:---:       |
|ERM|  resnet50                    | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune.sh)| - |
|ERM|  resnet50w2/w4 2x/4xresnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn_wide.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_wide.sh)| - |
|CAT| -                            | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cat_cifar_bn.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_cat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/two-satge_fine-tuning/rich_finetune.sh) |
|Distill| resnet50                 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/distill5.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_distill5.sh)| - |
|ERM|  resnet50                    | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune.sh)| -|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/rn50wide.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_wide.sh)|-|
|CAT| -                            | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/cat_inat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_cat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/rich_finetune.sh)|
|Distill| resnet50                 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/distill5.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_distill5.sh)|-|



## Supervised transfer learning (ViT)

### Download (Imagenet21k) pretrained & (ImageNet1k) finetuned ViT checkpoints according to [download_checkpoint.md](download_checkpoint.md)

The resulting folder structure looks like:

```
📦 RRL
 ┣ 📂checkpoints
 ┃ ┣ 📂supervised_pretrain
 ┃ ┃ ┣ 📂vit
 ┃ ┃ ┃ ┣📜 vitaugreg/imagenet21k/ViT-B_16.npz
 ┃ ┃ ┃ ┣📜 vitaugreg/imagenet21k/ViT-L_16.npz
 ┃ ┃ ┃ ┣📜 vit/imagenet21k/ViT-B_16.npz
 ┃ ┃ ┃ ┗📜 vit/imagenet21k/ViT-L_16.npz
 ┃ ┃ ┣📜 vitaugreg/imagenet21k/imagenet2012/ViT-L_16.npz
 ┃ ┃ ┣📜 vitaugreg/imagenet21k/imagenet2012/ViT-L_16.npz
 ┃ ┃ ┣📜 vit/imagenet21k/imagenet2012/ViT-L_16.npz
 ┃ ┃ ┣📜 vit/imagenet21k/imagenet2012/ViT-L_16.npz

```


## self-supervised transfer learning

### Download SWAV and SEER checkpoints according to [download_checkpoint.md](download_checkpoint.md)

The resulting folder structure looks like:

```
📦 RRL
 ┣ 📂checkpoints
 ┃ ┣ 📂self_supervised_pretrain
 ┃ ┃ ┣📜 swav_400ep_pretrain.pth.tar
 ┃ ┃ ┣📜 swav_RN50w2_400ep_pretrain.pth.tar
 ┃ ┃ ┣📜 swav_RN50w4_400ep_pretrain.pth.tar
 ┃ ┃ ┣📜 swav_RN50w5_400ep_pretrain.pth.tar
 ┃ ┃ ┣📜 swav_400ep_pretrain_seed5.pth.tar
 ┃ ┃ ┣📜 swav_400ep_pretrain_seed6.pth.tar
 ┃ ┃ ┣📜 swav_400ep_pretrain_seed7.pth.tar
 ┃ ┃ ┣📜 swav_400ep_pretrain_seed8.pth.tar
 ┃ ┃ ┣📜 seer_regnet32gf.pth
 ┃ ┃ ┣📜 seer_regnet64gf.pth
 ┃ ┃ ┣📜 seer_regnet128gf.pth
 ┃ ┃ ┣📜 seer_regnet256gf.pth
 ┃ ┃ ┣📜 seer_regnet32gf_finetuned.pth
 ┃ ┃ ┣📜 seer_regnet64gf_finetuned.pth
 ┃ ┃ ┣📜 seer_regnet128gf_finetuned.pth
 ┃ ┃ ┣📜 seer_regnet256gf_finetuned.pth
```


### Transfer by Linear Probing, Fine-Tuning, and Two-stage Fine-Tuning (SWAV pretrained ImageNet1k):


## Meta-learning & few-shots learning and Out-of-distribution generalization
If you are further interested in the rest few-shots learning and out-of-distribution generalization code, please let me know by leaving a comment. 


## Citation
If you find this code useful for your research, please consider citing our work:
```
@inproceedings{zhang2023learning,
  title={Learning useful representations for shifting tasks and distributions},
  author={Zhang, Jianyu and Bottou, L{\'e}on},
  booktitle={International Conference on Machine Learning},
  pages={40830--40850},
  year={2023},
  organization={PMLR}
}
```

