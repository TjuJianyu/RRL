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
|CAT| resnet50                     | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cat_cifar_bn.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_cat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/two-satge_fine-tuning/rich_finetune.sh) |
|Distill| resnet50                 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/distill5.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_distill5.sh)| - |
|ERM|  resnet50                    | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune.sh)| -|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/rn50wide.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_wide.sh)|-|
|CAT| resnet50                     | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/cat_inat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_cat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/rich_finetune.sh)|
|Distill| resnet50                 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/distill5.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_distill5.sh)|-|



<!-- 
|method  |   architecture  |   params  |   CIFAR10  |   CIFAR100  |   INAT18  |   CIFAR10  |   CIFAR100  |   INAT18|
|--------|--------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|ERM  |   RESNET50  |   23.5M  |   97.54  |   85.58  |   64.19  |   -  |   -  |   -|
|ERM  |   RESNET50W2  |   93.9M  |   97.76  |   87.13  |   66.72  |   -  |   -  |   -|
|ERM  |   RESNET50W4  |   375M  |   97.88  |   87.95  |   66.99  |   -  |   -  |   -|
|ERM  |   2×RESNET50  |   47M  |   97.39  |   85.77  |   62.57  |   -  |   -  |   -|
|ERM  |   4×RESNET50  |   94M  |   97.38  |   85.56  |   61.58  |   -  |   -  |   -|
|CAT2  |   2×RESNET50  |   47M  |   97.56  |   86.04  |   64.49  |   97.87  |   87.07  |   66.96|
|CAT4  |   4×RESNET50  |   94M  |   97.53  |   86.54  |   64.54  |   98.14  |   88.00  |   68.42|
|CAT5  |   5×RESNET50  |   118M  |   97.57  |   86.46  |   64.86  |   98.19  |   88.11  |   68.48|
|CAT10  |   10×RESNET50  |   235M  |   97.19  |   86.65  |   64.39  |   98.17  |   88.50  |   69.07|
|DISTILL5  |   RESNET50  |   23.5M  |   97.07  |   85.31  |   64.17  |   -  |   -  |   -|

 -->
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


### Transfer by Linear Probing

scripts/supervised_transfer/imagenet21k/imagenet/vit.sh
scripts/supervised_transfer/imagenet21k/imagenet/vitaugreg.sh
scripts/supervised_transfer/imagenet21k/imagenet/bcat2_vitaugreg.sh
scripts/supervised_transfer/imagenet21k/imagenet/cat_vit.sh

#### Transfer by two-stage fine-tuning

scripts/supervised_transfer/imagenet21k/imagenet/2ft_bcat2_vitagureg.sh
scripts/supervised_transfer/imagenet21k/imagenet/2ft_vit.sh


## self-supervised transfer learning

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
 ┃ ┃ 
```

### Download [SWAV](https://github.com/facebookresearch/swav) Imagenet1k pretrained checkpoints:
```
mkdir checkpoints/swav -p
wget https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar -O checkpoints/self_supervised_pretrain/swav_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w2_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w4_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w5_400ep_pretrain.pth.tar
```

#### Download more [SWAV](https://github.com/facebookresearch/swav) Imagenet1k pretrained ResNet50 checkpoint with different seeds:


To have mutiple SWAV Imagenet1k pretrained ResNet50 models with only different seeds, one can simply download our pretrained checkpoints from [here](https://drive.google.com/file/d/1D2DCInQKpqgqQC3dxtf5eOLD-dSrm2yN/view?usp=sharing) or train them from scratch according to [SWAV code](https://github.com/facebookresearch/swav/blob/main/scripts/swav_400ep_pretrain.sh). (NOTE: customize the ```--seed``` option.)



### SEER (Instagram-1B)
Pretrained and ImageNet finetuned checkpoints comes from [here](https://github.com/facebookresearch/vissl/tree/main/projects/SEER)
#### Download checkpoints

Download [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) Instagram-1B pretrained checkpoints:
```
mkdir checkpoints/seer -p
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch -O checkpoints/self_supervised_pretrain/seer_regnet32gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet64gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet128gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet256gf.pth
```


Download ImageNet1k finetuned [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) checkpoints:
```
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet32gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet64gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet128gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch -O checkpoints/self_supervised_pretrain/seer_regnet256gf_finetuned.pth
```


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

