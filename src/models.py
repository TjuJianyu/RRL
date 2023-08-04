from classy_vision.models import build_model as build_model 
from classy_vision.models import  RegNet as ClassyRegNet

import numpy as np 
import logging
from typing import List, Tuple
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .resnet_cifar import ResNet18 
from .visiontransformer.modeling import VisionTransformer, CONFIGS, np2th
import math

# copied from https://github.com/facebookresearch/vissl/blob/9551cfb490704ac151b2067b43ea595cda1adf4b/vissl/models/model_helpers.py#L38
def transform_model_input_data_type(model_input, input_type: str):
	"""
	Default model input follow RGB format. Based the model input specified,
	change the type. Supported types: RGB, BGR, LAB
	"""
	model_output = model_input
	# In case the model takes BGR input type, we convert the RGB to BGR
	if input_type == "bgr":
		model_output = model_input[:, [2, 1, 0], :, :]
	# In case of LAB image, we take only "L" channel as input. Split the data
	# along the channel dimension into [L, AB] and keep only L channel.
	if input_type == "lab":
		model_output = torch.split(model_input, [1, 2], dim=1)[0]
	return model_output

# copied from https://github.com/facebookresearch/vissl/
class Flatten(nn.Module):
	"""
	Flatten module attached in the model. It basically flattens the input tensor.
	"""

	def __init__(self, dim=-1):
		super(Flatten, self).__init__()
		self.dim = dim

	def forward(self, feat):
		"""
		flatten the input feat
		"""
		return torch.flatten(feat, start_dim=self.dim)

	def flops(self, x):
		"""
		number of floating point operations performed. 0 for this module.
		"""
		return 0


# modified from https://github.com/facebookresearch/vissl/
class RegNet(nn.Module):
	"""
	Wrapper for ClassyVision RegNet model so we can map layers into feature
	blocks to facilitate feature extraction and benchmarking at several layers.

	This model is defined on the fly from a RegNet base class and a configuration file.

	We follow the feature naming convention defined in the ResNet vissl trunk.
	"""

	def __init__(self, model_config, skip_pool=False):
		super().__init__()

		trunk_config = model_config
		if "name" in trunk_config:
			name = trunk_config["name"]
			if name == "anynet":
				model = build_model(trunk_config)
			else:
				logging.info(f"Building model: RegNet: {name}")
				model = build_model({"name": name})
		else:
			logging.info("Building model: RegNet from yaml config")
			model = ClassyRegNet.from_config(trunk_config)

		# Now map the models to the structure we want to expose for SSL tasks
		# The upstream RegNet model is made of :
		# - `stem`
		# - n x blocks in trunk_output, named `block1, block2, ..`

		# We're only interested in the stem and successive blocks
		# everything else is not picked up on purpose
		feature_blocks: List[Tuple[str, nn.Module]] = []

		# - get the stem
		feature_blocks.append(("conv1", model.stem))

		# - get all the feature blocks
		for k, v in model.trunk_output.named_children():
			assert k.startswith("block"), f"Unexpected layer name {k}"
			block_index = len(feature_blocks) + 1
			feature_blocks.append((f"res{block_index}", v))

		# - finally, add avgpool and flatten.
		if not skip_pool:
			feature_blocks.append(("avgpool", nn.AdaptiveAvgPool2d((1, 1))))
			feature_blocks.append(("flatten", Flatten(1)))

		#self._feature_blocks = nn.ModuleDict(feature_blocks)
		self._feature_blocks = nn.Sequential(OrderedDict(feature_blocks))
	def forward(self,x):
		model_input = transform_model_input_data_type(
				x, 'rgb'
			)
		output = self._feature_blocks(model_input)
		return output 

#https://github.com/facebookresearch/swav/blob/main/src/resnet50.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		groups=groups,
		bias=False,
		dilation=dilation,
	)

#https://github.com/facebookresearch/swav/blob/main/src/resnet50.py
def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# modified from https://github.com/facebookresearch/swav/blob/main/src/resnet50.py
class BasicBlock(nn.Module):
	expansion = 1
	__constants__ = ["downsample"]

	def __init__(
		self,
		inplanes,
		planes,
		stride=1,
		downsample=None,
		groups=1,
		base_width=64,
		dilation=1,
		norm_layer=None,
	):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError("BasicBlock only supports groups=1 and base_width=64")
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)

		return out

# modified from https://github.com/facebookresearch/swav/blob/main/src/resnet50.py
class Bottleneck(nn.Module):
	expansion = 4
	__constants__ = ["downsample"]

	def __init__(
		self,
		inplanes,
		planes,
		stride=1,
		downsample=None,
		groups=1,
		base_width=64,
		dilation=1,
		norm_layer=None,
	):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.0)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)


		out += identity
		out = self.relu(out)

		return out

# modified from https://github.com/facebookresearch/swav/blob/main/src/resnet50.py
class ResNet(nn.Module):
	def __init__(
			self,
			block,
			layers,
			zero_init_residual=False,
			groups=1,
			widen=1,
			width_per_group=64,
			replace_stride_with_dilation=None,
			norm_layer=None,
			normalize=False,
			output_dim=0,
			hidden_mlp=0,
			#nmb_prototypes=0,
			eval_mode=False,
			#shortcut=(None,None,None),
			fs = [None, None],
			smallimage=False,
	):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.fs = fs 
		self.eval_mode = eval_mode
		self.padding = nn.ConstantPad2d(1, 0.0)
		#self.shortcut = shortcut
		self.inplanes = width_per_group * widen
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError(
				"replace_stride_with_dilation should be None "
				"or a 3-element tuple, got {}".format(replace_stride_with_dilation)
			)
		self.groups = groups
		self.base_width = width_per_group

		# change padding 3 -> 2 compared to original torchvision code because added a padding layer
		num_out_filters = width_per_group * widen
		
		self.smallimage = smallimage
		if self.smallimage: # and not max pool
			self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)
		else:
			self.conv1 = nn.Conv2d(
				3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
			)
		self.bn1 = norm_layer(num_out_filters)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, num_out_filters, layers[0])
		num_out_filters *= 2
		
		self.layer2 = self._make_layer(
			block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
		)
		num_out_filters *= 2
		self.layer3 = self._make_layer(
			block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
		)
		num_out_filters *= 2
		self.layer4 = self._make_layer(
			block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
		)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		# normalize output features
		self.l2norm = normalize

		# projection head
		if output_dim == 0:
			self.projection_head = None
		elif hidden_mlp == 0:
			self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
		else:
			self.projection_head = nn.Sequential(
				nn.Linear(num_out_filters * block.expansion, hidden_mlp),
				nn.BatchNorm1d(hidden_mlp),
				nn.ReLU(inplace=True),
				nn.Linear(hidden_mlp, output_dim),
			)

		# prototype layer
		# self.prototypes = None
		# if isinstance(nmb_prototypes, list):
		# 	self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
		# elif nmb_prototypes > 0:
		# 	self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(
			block(
				self.inplanes,
				planes,
				stride,
				downsample,
				self.groups,
				self.base_width,
				previous_dilation,
				norm_layer,
			)
		)
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(
				block(
					self.inplanes,
					planes,
					groups=self.groups,
					base_width=self.base_width,
					dilation=self.dilation,
					norm_layer=norm_layer,
				)
			)

		return nn.Sequential(*layers)

	def forward_backbone(self, x):
		x = self.padding(x)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		if not self.smallimage:
			x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		if self.eval_mode:
			return x

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		if self.fs[0] is not None and self.fs[1] is not None: 
			return x[:,self.fs[0]:self.fs[1]]
		else:
			return x
		

	def forward_head(self, x):
		if self.projection_head is not None:
			x = self.projection_head(x)

		if self.l2norm:
			x = nn.functional.normalize(x, dim=1, p=2)

		# if self.prototypes is not None:
		# 	return x, self.prototypes(x)
		return x

	def forward(self, inputs):
		if not isinstance(inputs, list):
			inputs = [inputs]
		idx_crops = torch.cumsum(torch.unique_consecutive(
			torch.tensor([inp.shape[-1] for inp in inputs]),
			return_counts=True,
		)[1], 0)
		start_idx = 0
		for end_idx in idx_crops:
			_out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
			if start_idx == 0:
				output = _out
			else:
				output = torch.cat((output, _out))
			start_idx = end_idx
		return self.forward_head(output)


def resnet50(**kwargs):
	return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
	return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
	return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
	return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)


class ViT(VisionTransformer):
	def __init__(self, config, img_size=224, num_classes=1, vis=False):
		super(ViT, self).__init__(config=config, img_size=img_size,num_classes=num_classes, zero_head=True, vis=vis)
  
	def forward(self,x):
		rep, _ = self.transformer(x)
		return rep[:, 0]

class ModelWrapper(torch.nn.Module):
	def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
		super(ModelWrapper, self).__init__()
		self.model = model
		self.classification_head = torch.nn.Linear(feature_dim, num_classes)
		self.normalize = normalize
		if initial_weights is None:
			initial_weights = torch.zeros_like(self.classification_head.weight)
			torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
		self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
		self.classification_head.bias = torch.nn.Parameter(
			torch.zeros_like(self.classification_head.bias))

		# Note: modified. Get rid of the language part.
		if hasattr(self.model, 'transformer'):
			delattr(self.model, 'transformer')

	def forward(self, images, return_features=False):
		features = self.model.encode_image(images)
		#print(features.norm(dim=-1, keepdim=True).mean())
		if self.normalize:
			features = features / features.norm(dim=-1, keepdim=True)
			
		logits = self.classification_head(features)
		if return_features:
			return logits, features
		return logits

def get_model_from_sd(state_dict, base_model):
	feature_dim = state_dict['classification_head.weight'].shape[1]
	num_classes = state_dict['classification_head.weight'].shape[0]
	model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
	for p in model.parameters():
		p.data = p.data.float()
	model.load_state_dict(state_dict)
	model = model.cuda()
	devices = [x for x in range(torch.cuda.device_count())]
	return model 

def get_model(name, skip_pool=False, pretrain_path=None, img_dim=None):
	
	accept_models = ['regnet_y_32gf', 'regnet_y_64gf', 'regnet_y_128gf','regnet_y_256gf',\
					'resnet50', '2resnet50', '4resnet50', 'resnet50w2','resnet50w4','resnet50w5',\
					'resnet152',\
					'resnet18','resnet50w2','resnet50w4','2resnet18','4resnet18']
	
	assert name in accept_models, f'{name} not in  accept_models {accept_models}'
	
	# if name.lower().startswith('clip'):
	# 	clip_model_name = name[len('clip'):].lower()
	# 	if clip_model_name == 'vit-b/32':
	# 		model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
	# 	else:
	# 		raise NotImplementedError
	# 	state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
	# 	model = get_model_from_sd(state_dict, model)
	# 	model.classification_head = nn.Identity() 
	# 	feat_dim = 512
	# 	msg = f'clip {name} done' 
	# 	return model, msg, feat_dim

	if 'regnet' in name.lower():
		
		def model_loader(path, args=None):
			state_dict = torch.load(path, map_location = "cpu")
			state_dict = state_dict['classy_state_dict']['base_model']['model']['trunk']
			return state_dict, None 


		if name == 'regnet_y_32gf':
			model = RegNet({'name':'regnet_y_32gf'}, skip_pool=skip_pool)
			feat_dim = 3712
			
		elif name == 'regnet_y_64gf':
			model = RegNet({'name':'regnet_y_64gf'}, skip_pool=skip_pool)
			feat_dim = 4920
			
		elif name == 'regnet_y_128gf':
			model = RegNet({'name':'regnet_y_128gf'}, skip_pool=skip_pool)
			feat_dim = 7392
			
		elif name == 'regnet_y_256gf':
			model = RegNet({'depth': 27,
							'w_0': 640,
							'w_a': 230.83,
							'w_m': 2.53,
							'group_width': 373}, skip_pool=skip_pool)	 
			feat_dim = 10444
			
		else:
			raise NotImplementedError

		if pretrain_path is not None:
			state_dict, _ = model_loader(pretrain_path)
			msg = model.load_state_dict(state_dict, strict = False)
		else:
			msg = "model is initialized without loading weights"
		return model, msg, feat_dim

	elif 'resnet' in name.lower():

		def model_loader(path,args=None):
			state_dict = torch.load(path,map_location = 'cpu')
			
			state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
			new_state_dict = {}
			for key in state_dict:
				if key.startswith("module.model0."):
					new_state_dict[key.replace('module.model0.','')] = state_dict[key]
				elif key.startswith("module."):
					new_state_dict[key.replace('module.','')] = state_dict[key]
				else:
					new_state_dict[key]=state_dict[key]
			
			return new_state_dict, None

		if name.lower() == 'resnet50w5':
			feat_dim = 2048 * 5 
			model = resnet50w5(output_dim=0, eval_mode= skip_pool)

		elif name.lower() == 'resnet50w4':
			feat_dim = 2048 * 4
			model =  resnet50w4(output_dim=0, eval_mode= skip_pool)

		elif name.lower() == 'resnet50w2':
			feat_dim = 2048 * 2
			model = resnet50w2(output_dim=0, eval_mode= skip_pool)
		
		# elif name.lower() == 'resnet50w2_fs0':
		# 	feat_dim = 2048
		# 	model = resnet50w2(output_dim=0, eval_mode= skip_pool, fs=[0,feat_dim])
		# elif name.lower() == 'resnet50w2_fs1':
		# 	feat_dim = 2048
		# 	model = resnet50w2(output_dim=0, eval_mode= skip_pool, fs=[feat_dim, 2*feat_dim])

		# elif name.lower() == 'resnet50w4_fs0':
		# 	feat_dim = 2048
		# 	model =  resnet50w4(output_dim=0, eval_mode= skip_pool,fs=[0,feat_dim])

		# elif name.lower() == 'resnet50w4_fs1':
		# 	feat_dim = 2048
		# 	model =  resnet50w4(output_dim=0, eval_mode= skip_pool,fs=[feat_dim, 2*feat_dim])
		
		# elif name.lower() == 'resnet50w4_fs2':
		# 	feat_dim = 2048
		# 	model =  resnet50w4(output_dim=0, eval_mode= skip_pool,fs=[2*feat_dim, 3*feat_dim])
		
		# elif name.lower() == 'resnet50w4_fs3':
		# 	feat_dim = 2048
		# 	model =  resnet50w4(output_dim=0, eval_mode= skip_pool,fs=[3*feat_dim, 4*feat_dim])
			

		elif name.lower() == '4resnet50':
			feat_dim = 2048 * 4
			model = Kmodel([resnet50(output_dim=0, eval_mode= skip_pool),
							resnet50(output_dim=0, eval_mode= skip_pool),
							resnet50(output_dim=0, eval_mode= skip_pool),
							resnet50(output_dim=0, eval_mode= skip_pool)])

		elif name.lower() == '2resnet50':
			feat_dim = 2048 * 2
			model = Kmodel([resnet50(output_dim=0, eval_mode= skip_pool),
							resnet50(output_dim=0, eval_mode= skip_pool)])

		elif name.lower() == 'resnet50':
			feat_dim = 2048 
			model = resnet50(output_dim=0, eval_mode= skip_pool)


		elif name.lower() == 'resnet152':
			feat_dim = 2048 
			model = models.resnet152()
			model.fc = nn.Identity()


		elif 'resnet18' in name.lower():
			def model_loader(path,args=None):
				state_dict = torch.load(path,map_location = 'cpu')
				state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
				new_state_dict = {}
				for key in state_dict:
					#print(key)
					if key.startswith("backbone."):
						new_state_dict[key.replace('backbone.','')] = state_dict[key]
					elif key.startswith('module.model0.'):
						new_state_dict[key.replace('module.model0.','')] = state_dict[key]
					else:
						new_state_dict[key] = state_dict[key]
				return new_state_dict, None  
			
			if name.lower() == 'resnet18x2' or name.lower() == 'resnet18w2':
				feat_dim = 512 * 2 
				model = ResNet18(width=2, avg_pool2d= (not skip_pool))
				
			if name.lower() == 'resnet18x4' or name.lower() == 'resnet18w4':
				feat_dim = 512 * 4
				model = ResNet18(width=4, avg_pool2d= (not skip_pool))
				
			if name.lower() == '2resnet18':
				feat_dim = 512 * 2 
				net1=ResNet18(width=1, avg_pool2d=(not skip_pool))
				net2=ResNet18(width=1, avg_pool2d=(not skip_pool))
				model = Kmodel([net1,net2])
				
			if name.lower() == '4resnet18':
				feat_dim = 512 * 4 
				model = Kmodel([ResNet18(width=1, avg_pool2d= ~skip_pool),
								ResNet18(width=1, avg_pool2d= ~skip_pool),
								ResNet18(width=1, avg_pool2d= ~skip_pool),
								ResNet18(width=1, avg_pool2d= ~skip_pool)])
			
		else:
			raise NotImplementedError

		# for val in model.parameters():
		# 	print(val)
		if pretrain_path is not None:
			state_dict, _ = model_loader(pretrain_path)

			msg = model.load_state_dict(state_dict, strict = False)
		else:
			msg = "model is initialized without loading weights"
		return model, msg, feat_dim
		

		
	elif 'vit' in name.lower(): 
		
		prenorm = True if name.startswith('prenorm') else False 
		modelname= name.lower().replace('prenorm','')
		
		names = {key.lower():key for key in CONFIGS}
		if modelname in names: 
			config = CONFIGS[names[modelname]]
		else:
			raise NotImplementedError

		#model = VisionTransformer(config, img_dim, zero_head=True, num_classes=1)
		model = ViT(config, img_dim)
		feat_dim = config.hidden_size

		if pretrain_path is not None:
			model.load_from(np.load(pretrain_path))
			msg = "model is initialized and loaded"
		else:
			msg = "model is initialized without loading weights"

		if prenorm:
			model.transformer.encoder.encoder_norm = nn.Identity()
		model.head = nn.Identity()
		print(model)
		return model, msg, feat_dim

	else:
		raise NotImplementedError

def load_classifier(name,classifier, args):


	if name == 'cat_weights':

		weight, bias = [], 0 	
		for path in args.pretrained:
			if path[-3:] == 'npz': # numpy npz checkpoint 
				checkpoint = np.load(path)
				w = np2th(checkpoint["head/kernel"]).t()
				weight.append(w)
				bias += np2th(checkpoint["head/bias"]).t()
				print(f'(cat) classifier weights loaded, {w.shape}')
			else:
				state_dict = torch.load(path, map_location='cpu')
				if 'classy_state_dict' in state_dict:
					head = state_dict['classy_state_dict']['base_model']['model']['heads']
					weight.append(head['0.clf.0.weight'])
					bias += head['0.clf.0.bias']
					
				elif 'state_dict' in state_dict:

					head = state_dict['state_dict']
					keyworks = ['module.fc', 'fc', 'module.classifier', 'module.classifier.linear']

					find = False 
					for key in keyworks:
						if key+'.weight' in head:
							weight.append(head[key+'.weight'])
							bias += head[key+'.bias']
							find = True 
							break 

					if not find:
						for key in head:
							print(key)
						raise NotImplementedError 

				else:
					raise NotImplementedError

		weight = torch.cat(weight, dim= 1)
		weight /= len(args.pretrained)
		bias /= len(args.pretrained)
		classifier.linear.weight.data = weight 
		classifier.linear.bias.data = bias 

		
		return classifier


	elif name == 'dumped_weights':
		if type(args.headpretrained) ==  str:
			if args.headpretrained[-3:] == 'npz':
				checkpoint = np.load(path)
				classifier.weight.copy_(np2th(checkpoint["head/kernel"]).t())
				classifier.bias.copy_(np2th(checkpoint["head/bias"]).t())
				msg = 'classifier dumped weights loaded'
			else:
				state_dict = torch.load(args.headpretrained, map_location='cpu')['state_dict']
				state_dict = {key.replace('module.classifier.',''): state_dict[key] for key in state_dict}

				msg = classifier.load_state_dict(state_dict, strict=False)
			print(msg)
		else: 
			weight, bias = [], 0 	
			for path in args.headpretrained: 
				if path[-3:] == 'npz': # numpy npz checkpoint 
					checkpoint = np.load(path)
					weight.append(np2th(checkpoint["head/kernel"]).t())
					bias += np2th(checkpoint["head/bias"]).t()
				else:
					head = torch.load(path, map_location='cpu')['state_dict']
					head = {key.replace('module.classifier.',''): head[key] for key in head}
					#ƒprint(head)
					weight.append(head['linear.weight'] )
					bias += head['linear.bias']

			weight = torch.cat(weight, dim= 1)
			weight /= len(args.headpretrained)
			bias /= len(args.headpretrained)
			classifier.linear.weight.data = weight 
			classifier.linear.bias.data = bias 


		return classifier
	
	elif name == 'none':
		return classifier
	elif name == 'normal':
		classifier.linear.weight.data.normal_(mean=0.0, std=0.01)
		classifier.linear.bias.data.zero_()
		return classifier
	# elif name.startswith('normal_fanin_scale'):
	# 	scale = float(name.split('_')[3])
	# 	in_dim = classifier.linear.weight.data.shape[1]
	# 	classifier.linear.weight.data.normal_(mean=0.0, std=np.sqrt(1/in_dim))
	# 	classifier.linear.bias.data.zero_()
	# 	classifier.linear.weight.data *= scale 

	# 	print(f'classifier weights normal: scale is {scale}, in_dim is {in_dim}')

	# 	return classifier
	# elif name.startswith('uniform_fanin_scale'):
	# 	scale = float(name.split('_')[3])
	# 	in_dim = classifier.linear.weight.data.shape[1]
	# 	k=np.sqrt(1/in_dim)
	# 	classifier.linear.weight.data.uniform_(-k,k)
	# 	classifier.linear.bias.data.uniform_(-k,k)
	# 	classifier.linear.weight.data *= scale 

	# 	print(f'classifier weights uniform: scale is {scale}, in_dim is {in_dim}')

	else:
		print(name)
		raise NotImplementedError


class Kmodel(nn.Module):
	def __init__(self, models, classifier=None, num_labels = 1000):
		super(Kmodel, self).__init__()
		#self.feat_dim = feat_dim
		self.models = models 
		self.classifier = classifier 
		for i in range(len(self.models)):
			setattr(self, 'model%d' % i, self.models[i])

	def get_feature(self,x):
		rep =  torch.cat([model(x) for model in self.models],dim=1)
		return rep
	def get_logits(self,x):
		if self.classifier is not None:
			return self.classifier(x)
		else:
			return x 

	def forward(self,x):

		rep =  torch.cat([model(x) for model in self.models],dim=1)

		if self.classifier is not None:
			return self.classifier(rep)
		else:
			return rep 

class RegLog(nn.Module):
	"""Creates logistic regression on top of frozen features"""
	def __init__(self, num_labels, feat_dim, use_bn=True, reinit_head=True):
		super(RegLog, self).__init__()
		
		self.bn = None 
		if use_bn:
			self.bn = nn.BatchNorm1d(feat_dim)

		self.linear = nn.Linear(feat_dim, num_labels)
		if reinit_head:
			print('reinit head weights by gaussian(0, 0.01). To be abandoned.')
			self.linear.weight.data.normal_(mean=0.0, std=0.01)
			self.linear.bias.data.zero_()

	def forward(self, x):
		# optional BN
		if self.bn is not None:
			x = self.bn(x)
		x = x.view(x.size(0), -1)
		return self.linear(x)


def get_classifier(classifier_arch, nclass, feat_dims, logger, args):
	
	# calculate classifier input dimension
	if args.richway == 'cat':
		total_feat_dim = int(np.sum(feat_dims))
	else:
		assert np.std(feat_dims) == 0
		total_feat_dim = int(feat_dims[0])


	if classifier_arch == 'linear':
		classifier = RegLog(nclass, total_feat_dim, args.use_bn, reinit_head=False)
		classifier = load_classifier(args.headinit, classifier, args)
	else:
		raise NotImplementedError

	return classifier
		
def modelfusion(richway, models, classifier, args):

	assert 'exp_mode' in args.__dict__ and 'sync_bn' in args.__dict__

	mappings = {'cat':Kmodel}	

	if args.exp_mode == 'finetune':
		model = nn.Identity()
		classifier = mappings[richway](models, classifier)
	elif args.exp_mode == 'lineareval':
		model = mappings[richway](models)

	if args.sync_bn:
		classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
	
	return model, classifier

