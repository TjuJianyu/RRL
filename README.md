## Rich Representation Learning.
### Official code for [learning useful representations for shifting tasks and distributions](https://arxiv.org/abs/2212.07346)


## requirements
portalocker
pyyaml

## datasets

#### ImageNet:
Download [ImageNet](https://www.image-net.org/download.php), extract and move it under `data/` folder with name `data/imagenet`.

#### Inaturalist18
Download [Inaturalist18](https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz) dataset, extract then move it under `data/` folder with name `data/inaturalist18`



## Supervised transfer
### ImageNet1k supervise pretraining
Pretrain resnet50 10 times on ImageNet1k with different random seeds as [script](scripts/supervised_transfer/supervised_pretrain/resnet50.sh). Alternatively, we provide 10 such checkpoints at [here](https://drive.google.com/file/d/1puDJCfUdexV7jc2QDtzT3GIV6bK_a5DS/view?usp=sharing).


### transfer

### Download vision transformer checkpoints
### transfer 


## self-supervised transfer

### SWAV
#### Download [SWAV](https://github.com/facebookresearch/swav) Imagenet1k pretrained checkpoints:
```
mkdir checkpoints/swav -p
wget https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar -O checkpoints/swav/swav_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar -O  checkpoints/swav/swav_RN50w2_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar -O  checkpoints/swav/swav_RN50w4_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar -O  checkpoints/swav/swav_RN50w5_400ep_pretrain.pth.tar
```

#### Pretrain SWAV mutiple times with different seeds
To have mutiple SWAV checkpoints with only different seeds, we need to pretrain SWAV mutiple times with different seeds.
Here we pretrain SWAV ResNet50 on ImageNet1k with different seeds using [code](https://github.com/facebookresearch/swav/blob/main/scripts/swav_400ep_pretrain.sh). (NOTE: customize the ```--seed``` option.)

[TODO can we public our pretrained checkpoints?](https://drive.google.com/file/d/1D2DCInQKpqgqQC3dxtf5eOLD-dSrm2yN/view?usp=sharing)


### SEER (Instagram-1B)
Pretrained and ImageNet finetuned checkpoints comes from [here](https://github.com/facebookresearch/vissl/tree/main/projects/SEER)
#### Download checkpoints

Download [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) Instagram-1B pretrained checkpoints:
```
mkdir checkpoints/seer -p
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch -O checkpoints/seer/seer_regnet32gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch -O checkpoints/seer/seer_regnet64gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/seer/seer_regnet128gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/seer/seer_regnet256gf.pth
```


Download ImageNet1k finetuned [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) checkpoints:
```
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/seer/seer_regnet32gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/seer/seer_regnet64gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/seer/seer_regnet128gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch -O checkpoints/seer/seer_regnet256gf_finetuned.pth
```


### transfer


## few-shot learning and meta-learning

## OOD

