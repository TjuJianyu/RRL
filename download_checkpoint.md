### Download (ImageNet1k) pretrained checkpoints:

| architecture| N repeats | url    | args |
| :---:       |    :----: | :---:  | :---:|
| resnet50    | 10      | [model](https://drive.google.com/file/d/1puDJCfUdexV7jc2QDtzT3GIV6bK_a5DS/view?usp=sharing)  | [script](scripts/supervised_transfer/imagenet/supervised_pretrain/resnet50.sh)|
| resnet50w2  | 1       | [model](https://drive.google.com/file/d/1yxpGox1on8EG-bgh5m96P-HmFdF1FqKV/view?usp=sharing)  | [script](scripts/supervised_transfer/imagenet/supervised_pretrain/resnet50_wide.sh)|
| resnet50w4  | 1       | [model](https://drive.google.com/file/d/1BMCdWbRp4nUxRQwKux-_BEQS_5TKC2h6/view?usp=sharing)  | [script](scripts/supervised_transfer/imagenet/supervised_pretrain/resnet50_wide.sh)|   
| 2resnet50   | 1       | [model](https://drive.google.com/file/d/1vC5es1ysSSZOEhkKQWBafjRyLR_oFPgl/view?usp=sharing)  | [script](scripts/supervised_transfer/imagenet/supervised_pretrain/resnet50_wide.sh)|
| 4resnet50   | 1       | [model](https://drive.google.com/file/d/1J3adr3hepZZXyLcncduBi3v6PZLPAEW5/view?usp=sharing)  | [script](scripts/supervised_transfer/imagenet/supervised_pretrain/resnet50_wide.sh)|

#### Download distilled ResNet50 checkpoint from [here](https://drive.google.com/file/d/1iS82WpEWaTqU6I1qbzEB64mttIp1dYDz/view?usp=sharing)


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


### Download [SWAV](https://github.com/facebookresearch/swav) Imagenet1k pretrained checkpoints:
```
mkdir checkpoints/swav -p
wget https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar -O checkpoints/self_supervised_pretrain/swav_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w2_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w4_400ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar -O  checkpoints/self_supervised_pretrain/swav_RN50w5_400ep_pretrain.pth.tar
```

#### Download more SWAV Imagenet1k pretrained ResNet50 checkpoint with different seeds:


To have mutiple SWAV Imagenet1k pretrained ResNet50 models with only different seeds, one can simply:
- download our pretrained checkpoints from [here](https://drive.google.com/file/d/1D2DCInQKpqgqQC3dxtf5eOLD-dSrm2yN/view?usp=sharing) or 
- train them from scratch according to [SWAV code](https://github.com/facebookresearch/swav/blob/main/scripts/swav_400ep_pretrain.sh). (NOTE: customize the ```--seed``` option.)



#### Download [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) Instagram-1B pretrained checkpoints

```
mkdir checkpoints/seer -p
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch -O checkpoints/self_supervised_pretrain/seer_regnet32gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet64gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet128gf.pth
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch -O checkpoints/self_supervised_pretrain/seer_regnet256gf.pth
```

#### Download SEER ImageNet1k finetuned checkpoints:

```
wget https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet32gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet64gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch -O checkpoints/self_supervised_pretrain/seer_regnet128gf_finetuned.pth
https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch -O checkpoints/self_supervised_pretrain/seer_regnet256gf_finetuned.pth
```