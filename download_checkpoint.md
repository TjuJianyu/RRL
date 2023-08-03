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