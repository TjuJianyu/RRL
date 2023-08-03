def model_configs(tag, args):

	# swav 
	if tag == 'swav_resnet50':
		args.arch = ['resnet50']
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar']
	
	elif tag == 'swav_resnet50w2':
		args.arch = ['resnet50w2']
		args.pretrained = ['../swav/swav_RN50w2_400ep_pretrain.pth.tar']
	
	elif tag == 'swav_resnet50w4':
		args.arch = ['resnet50w4']
		args.pretrained = ['../swav/swav_RN50w4_400ep_pretrain.pth.tar']

	elif tag == 'swav_resnet50w5':
		args.arch = ['resnet50w5']
		args.pretrained = ['../swav/swav_RN50w5_400ep_pretrain.pth.tar']
	

	elif tag  in ['swav_cat_5', 'swav_resnet50_cat_5']:
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50','resnet50']
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run8/checkpoint.pth.tar',  ]
	
	elif tag == 'swav_resnet50_cat_7':
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50','resnet50','resnet50', 'resnet50', ]
		
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run8/checkpoint.pth.tar', 
						'../swav/experiments/swav_400ep_pretrain_run10/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run11/checkpoint.pth.tar'  ]
	
	elif tag == 'swav_resnet50_cat_9':
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50','resnet50','resnet50', 'resnet50', 'resnet50', 'resnet50']
		
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run8/checkpoint.pth.tar', 
						'../swav/experiments/swav_400ep_pretrain_run10/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run11/checkpoint.pth.tar', 
						'../swav/experiments/swav_400ep_pretrain_run12/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run13/checkpoint.pth.tar', ]
	
	# cat swav
	elif tag == 'swav_cat_2' or tag =='swav_resnet50_cat_2':
		args.arch = ['resnet50', 'resnet50']

		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar' ]

	elif tag == 'swav_cat_4' or tag == 'swav_resnet50_cat_4':
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50']
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
						'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar', ]
	elif tag == 'swav_cat_7':
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50w2', 'resnet50w4' ]
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run8/checkpoint.pth.tar',
							'../swav/swav_RN50w2_400ep_pretrain.pth.tar',
							'../swav/swav_RN50w4_400ep_pretrain.pth.tar' ]
	elif tag == 'swav_cat_8':
		args.arch = ['resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50w2', 'resnet50w4','resnet50w5' ]
		args.pretrained = ['../swav/swav_400ep_pretrain.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run5/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run6/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run7/checkpoint.pth.tar',
							'../swav/experiments/swav_400ep_pretrain_run8/checkpoint.pth.tar',
							'../swav/swav_RN50w2_400ep_pretrain.pth.tar',
							'../swav/swav_RN50w4_400ep_pretrain.pth.tar',
							'../swav/swav_RN50w5_400ep_pretrain.pth.tar', ]


	elif tag in ['seer_32gf', 'seer_64gf', 'seer_128gf','seer_256gf', 'seer_3B', 'seer_10B']:
		_ , size = tag.split('_')
		args.arch = ['regnet_y_%s' % size]
		args.pretrained = ['../SEER/checkpoints/seer_regnet%s.pth' % size]

	elif tag == 'seer_32_64':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf']
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf.pth',
							'../SEER/checkpoints/seer_regnet64gf.pth']

	elif tag == 'seer_32_64_128':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf', 'regnet_y_128gf']
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf.pth',
							'../SEER/checkpoints/seer_regnet64gf.pth',
							'../SEER/checkpoints/seer_regnet128gf.pth']

	elif tag == 'seer_32_64_128_256':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf', 'regnet_y_128gf', 'regnet_y_256gf' ]
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf.pth',
							'../SEER/checkpoints/seer_regnet64gf.pth',
							'../SEER/checkpoints/seer_regnet128gf.pth',
							'../SEER/checkpoints/seer_regnet256gf.pth']

	elif tag == 'seer_32_64_ft':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf']
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet64gf_finetuned.pth']
	elif tag == 'seer_32_64_128_ft':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf','regnet_y_128gf']
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet64gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet128gf_finetuned.pth']
	elif tag == 'seer_32_64_128_256_ft':
		args.arch = ['regnet_y_32gf', 'regnet_y_64gf','regnet_y_128gf', 'regnet_y_256gf']
		args.pretrained = ['../SEER/checkpoints/seer_regnet32gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet64gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet128gf_finetuned.pth',
							'../SEER/checkpoints/seer_regnet256gf_finetuned.pth']
	
	elif tag[:len('simsiam_cifar10_resnet18_cat')] == 'simsiam_cifar10_resnet18_cat':
		k = int(tag[len('simsiam_cifar10_resnet18_cat'):])
		args.arch = ['resnet18'] * k 
		args.pretrained = ['results/simsiam_cifar10/cifar10_simsiam_checkpoints/resnet18/run%d/ckpt_epoch_800_1.pth' % i for i in range(k)]

	elif 'simsiam' in tag:
		model = tag.split('_')[1]
		k = int(tag.split('_')[2])

		args.arch = [model] * k 
		args.pretrained = ['results/simsiam_cifar10/cifar10_simsiam_checkpoints/%s/run%d/ckpt_epoch_800_1.pth' % (model, i) for i in range(k) ]
	elif 'supervisedimagenet' in tag:
		model = tag.split('_')[1]
		k = int(tag.split('_')[2])

		args.arch = [model] * k 
		if model == 'resnet50':
			args.pretrained = ['results/supervised/imagenet/run%i/checkpoint.pth.tar' % (i) for i in range(k) ]
		else:
			args.pretrained = ['results/supervised/imagenet/%s/run%i/checkpoint.pth.tar' % (model, i) for i in range(k) ]

	elif 'supervisedboostimagenet' in tag:
		
		model = tag.split('_')[1]
		k = int(tag.split('_')[2])

		args.arch = [model] * k 
		if model == 'resnet50':
			args.pretrained = ['results/supervised/imagenet/run0/checkpoint.pth.tar'] + ['results/supervised/imagenet/run0/boost%d/checkpoint.pth.tar' % i for i in range(1,k)]
			
		else:
			raise NotImplementedError

	
	else:

		raise NotImplementedError

	return args 





