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

<!-- 
For few-shot learning / meta-learning, CUB and miniImagenet are tested. 

For out-of-distribution robustness, Camlyon17 are testd. 
 -->

## Supervised transfer learning (ResNet)


### Download (ImageNet1k) pretrained checkpoints:

You can download pretrained checkpoints either:
- by running ```python tools download.py``` or
- by hand according to [download_checkpoint.md](download_checkpoint.md)


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


### Transfer by Linear Probing

|method|architecture| target task |args|
|:---:|:---:       |:---:        |:---:       |
|ERM|  resnet50  | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn.sh)|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn_wide.sh)|
|CAT| resnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cat_cifar_bn.sh)|
|Distill| resnet50| Cifar10/Cifar100| |
|ERM|  resnet50  | Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat.sh)|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Inaturalist18||
|CAT| resnet50 | Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/cat_inat.sh)|
|Distill| resnet50| Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat_synt.sh) |


### Transfer by fine tuning and two-stage fine-tuning

|method|architecture| target task |args|
|:---:|:---:       |:---:        |:---:       |
|ERM|  resnet50  | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune.sh)|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune.sh)|
|CAT| resnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cat_cifar_bn.sh)|
|Distill| resnet50| Cifar10/Cifar100| |
|ERM|  resnet50  | Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat.sh)|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Inaturalist18||
|CAT| resnet50 | Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/cat_inat.sh)|
|Distill| resnet50| Inaturalist18|[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat_synt.sh) |



## Supervised transfer learning (ViT)

### Download (Imagenet21k) pretrained & (ImageNet1k) finetuned checkpoints:

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

#### Download VIT pretrained [checkpoints](https://github.com/google-research/vision_transformer):
```
mkdir checkpoints/vitaugreg/imagenet21k -p
wget https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz -O checkpoints/supervised_pretrain/vitaugreg/imagenet21k/ViT-L_16.npz
wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz -O checkpoints/supervised_pretrain/vitaugreg/imagenet21k/ViT-B_16.npz

mkdir checkpoints/vit/imagenet21k -p
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz -O checkpoints/supervised_pretrain/vit/imagenet21k/ViT-L_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -O checkpoints/supervised_pretrain/vit/imagenet21k/ViT-B_16.npz

```

#### Download VIT (Imagenet1k) fine-tuned [checkpoints](https://github.com/google-research/vision_transformer):
```

mkdir checkpoints/vitaugreg/imagenet21k/imagenet2012 -p
wget https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz -O checkpoints/vitaugreg/imagenet21k/imagenet2012/ViT-L_16.npz
wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz -O checkpoints/vitaugreg/imagenet21k/imagenet2012/ViT-B_16.npz

mkdir checkpoints/vit/imagenet21k/imagenet2012 -p
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz -O checkpoints/vit/imagenet21k/imagenet2012/ViT-L_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz -O checkpoints/vit/imagenet21k/imagenet2012/ViT-B_16.npz

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


### transfer
#### Linear probing


#### fine-tuning and two-stage fine-tuning

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

