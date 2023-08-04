## learning useful representations for shifting tasks and distributions

Official Pytorch implementation of [paper](https://arxiv.org/abs/2212.07346)

[Jianyu Zhang](https://www.jianyuzhang.com/),  [Léon Bottou](https://leon.bottou.org/)


<p align="center" width="500">
  <image src='figures/story.png'/>
</p>


## Requirements

- python==3.7
- torch>=1.13.1  
- torchvision>=0.14.1
- pyyaml==6.0
- classy-vision==0.6.0

## Datasets

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

Transfer the learned representation (on ImageNet1k) to Cifar10, Cifar100, and Inaturalist18 by:
- **Linear Probing**: concatenate these representation and learn a big linear classifier on top.
- **(Normal) Fine tuning**: concatenate pretrained representations then fine tuning all weights. 
- **(Two-stage) Fine tuning**: fine-tune each pretrained representation on target tasks separately, then concatenate the representation and apply linear probing. 

The following table provides scripts for these transfer learning experiments:

|method|architecture| target task |linear probing| fine-tuning | two-stage fine-tuning |
|:---:|:---:       |:---:        |:---:       |:---:       |:---:       |
|ERM|  resnet50                    | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune.sh)| - |
|ERM|  resnet50w2/w4 2x/4xresnet50 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cifar_bn_wide.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_wide.sh)| - |
|CAT| -                            | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/cat_cifar_bn.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_cat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/two-stage_fine-tuning/rich_finetune.sh) |
|Distill| resnet50                 | Cifar10/Cifar100|[scripts](scripts/supervised_transfer/imagenet/to_cifar/linear_probing/distill5.sh)| [scripts](scripts/supervised_transfer/imagenet/to_cifar/fine-tuning/finetune_distill5.sh)| - |
|ERM|  resnet50                    | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/inat.sh)| [scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune.sh)| -|
|ERM|  resnet50w2/w4 2x/4xresnet50 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/rn50wide.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_wide.sh)|-|
|CAT| -                            | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/cat_inat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_cat.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/two-stage_fine-tuning/rich_finetune.sh)|
|Distill| resnet50                 | Inaturalist18   |[scripts](scripts/supervised_transfer/imagenet/to_inat18/linear_probing/distill5.sh)|[scripts](scripts/supervised_transfer/imagenet/to_inat18/fine-tuning/finetune_distill5.sh)|-|

<p align="center">
<em>
Tab1: transfer learning experiments scripts.
</em>
</p>
 
The following figure shows (focus on solid curves) the transfer learning performance of different representations (ERM / CAT / Distill) and transfer methods (pinear probing / fine-tuning / two-stage fine-tuning). 

<p align="center">
  <image src='figures/imagenet_sl_tf_v3.png'/>
</p>

<p align="center">
<em>
  Fig1: Supervised transfer learning from ImageNet to Inat18, Cifar100, and Cifar10. The top row shows the superior linear
probing performance of the CATn networks (blue, “cat”). The bottom row shows the performance of fine-tuned CATn, which is poor with
normal fine-tuning (gray, “[init]cat”) and excellent for two-stage fine tuning (blue, “[2ft]cat”). DISTILLn (pink, “distill”) representation is obtained by distilling CATn into one ResNet50.
</em>
</p>


 
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

With the same experiment protocol as Tab1, we can have the following transfer learning curves with Vision Transformer: 

<p align="center">
  <image src='figures/vit_tf_v3.png' width="500"/>
</p>

<p align="center">
<em>
  Fig2: 
</em>
</p>


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

With the same experiment protocol as Tab1, we can have the following self-supervised transfer learning curves: 

<p align="center">
  <image src='figures/ssl_tf.png'/>
</p>

<p align="center">
<em>
  Fig2: Self-supervised transfer learning with SWAV trained on unlabeled ImageNet(1K) (top row) and with SEER on Instagram1B 
(bottom row). The constructed rich representation, CATn, yields the best linear probing performance (“cat” and “catsub”) for supervised
ImageNet, INAT18, CIFAR100, and CIFAR10 target tasks. The two-stage fine-tuning (“[2ft]cat”) matches equivalently sized baseline
models (“[init]wide” and “[init]wide&deep”), but with much easier training. The sub-networks of CAT5 (and CAT2) in SWAV hold the
same architecture
</em>
</p>


<!-- ### Transfer by Linear Probing, Fine-Tuning, and Two-stage Fine-Tuning (SWAV pretrained ImageNet1k): -->


## Meta-learning & few-shots learning and Out-of-distribution generalization

<p align="center">
  <image src='figures/meta_learning_full_v4.png'  width="500"/>
</p>

<p align="center">
<em>
  Fig3: Few-shot learning performance on MINIIMAGENET and
CUB. Four common few-shot learning algorithms are shown in
red (results from Chen et al. (2019)(https://arxiv.org/abs/1904.04232)). Two supervised transfer
methods, with either a linear classifier (BASELINE) or cosine-
based classifier (BASELINE++) are shown in blue. The DISTILL
and CAT results, with a cosine-base classifier, are respectively
shown in orange and gray. The CAT5-S and DISTILL5-S results
were obtained using five snapshots taken during a single training
episode with a relatively high step size. The dark blue line shows
the best individual snapshot. Standard deviations over five repeats
are reported.
</em>
</p>
<p align="center">

  <image src='figures/ood_general.png'  width="500"/>
</p>

<p align="center">
<em>
  Fig4: Test accuracy on the CAMELYON17 dataset with
DENSENET121. We compare various initialization (ERM, CATn,
DISTILLn, and Bonsai(https://arxiv.org/pdf/2203.15516.pdf)) for two algorithms VREX and ERM
using either the IID or OOD hyperparameter tuning method. The
standard deviations over 5 runs are reported.
</em>
</p>


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

