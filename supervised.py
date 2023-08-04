import argparse
import os
import time
from logging import getLogger
import warnings

import numpy as np 
from tqdm import tqdm
import yaml 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.distributed as dist
import torch.autograd as autograd

from src.utils import (
	bool_flag,
	initialize_exp,
	restart_from_checkpoint,
	fix_random_seeds,
	AverageMeter,
	init_distributed_mode,
	accuracy,
	add_slurm_params,
	get_dataloader,
	optimizer_config,
)

from src.models import get_model, get_classifier, modelfusion
from src.datasets import get_dataset 
#from src.configs import model_configs



logger = getLogger()


def main(args):

	global best_acc

	tags = yaml.load(open('configs/pretrained_checkpoints.yaml'), Loader=yaml.FullLoader)

	if args.tag is not None and args.tag in tags:
		for key in tags[args.tag]:
			print(key)
			setattr(args, key, tags[args.tag][key])

	# distributed training environments and seeds
	init_distributed_mode(args)
	fix_random_seeds(args.seed)
	
	# amd gpu cards environment variables
	os.environ['MIOPEN_USER_DB_PATH']=os.path.join(args.dump_path, 'amd/rank_%d' % args.rank)
	os.environ['MIOPEN_FIND_MODE']='2'

	# initialize logger ... 
	logger, training_stats = initialize_exp( args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val")

	# build data
	train_dataset, val_dataset, datamsg = get_dataset(args.data_name, args.tf_name, args)

	# build dataloaders 
	train_loader, val_loader, additional_loaders = get_dataloader(train_dataset, val_dataset, datamsg, args)
	logger.info("Building data done")


	## build trunk and load weights 
	total_feat_dim, feat_dims, models = 0, [], []

	for i in range(len(args.arch)):
	
		per_model, msg, feat_dim = get_model(args.arch[i], skip_pool=args.skip_pool, \
		pretrain_path = None if len(args.pretrained)==0 else args.pretrained[i], img_dim=datamsg['img_dim'])
		#fix1st_pretrain_path = args.fix1st_pretrained ) #e.g. 'regnet_y_32gf'
		logger.info("Load pretrained model with msg: {}".format(msg))

		feat_dims.append(feat_dim)
		models.append(per_model)

	
	#build classifier
	classifier = get_classifier(args.classifier, datamsg['nclass'], feat_dims, logger, args)

	#print(classifier.linear.weight.data)
	logger.info('classifier {}'.format(classifier))
	
	# model to gpu
	device = torch.device("cuda:" + str(args.gpu_to_work_on))

	# model is either Identity or backbone. only classifier is trainable
	model, classifier = modelfusion(args.richway, models, classifier, args)
	model, classifier = model.to(device), classifier.to(device)

	classifier = nn.parallel.DistributedDataParallel(
		classifier,
		device_ids=[args.gpu_to_work_on],
	)
	
	optimizer = optimizer_config(classifier, args, logger, \
		head_reg = lambda x: True if args.exp_mode in ['lineareval','biaslineareval'] else lambda x: 'classifier' in x )
	logger.info('optimizer {}'.format(optimizer))
	



	# set scheduler
	if args.scheduler_type == "step":
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer, args.decay_epochs, gamma=args.gamma
		)
	elif args.scheduler_type == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, args.epochs, eta_min=args.final_lr
		)
	logger.info('lr scheduler {}'.format(scheduler))




	# Optionally resume from a checkpoint
	to_restore = {"epoch": 0, "best_acc": 0.}
	
	if 'save' in args.mode:
		restart_from_checkpoint(
		os.path.join(args.dump_path, "checkpoint.pth.tar"),
		run_variables=to_restore,
		state_dict=classifier,
		)
		start_epoch = to_restore["epoch"]
		best_acc = to_restore["best_acc"]
	else:
		restart_from_checkpoint(
			os.path.join(args.dump_path, "checkpoint.pth.tar"),
			run_variables=to_restore,
			state_dict=classifier,
			optimizer=optimizer,
			scheduler=scheduler,
		)
		start_epoch = to_restore["epoch"]
		best_acc = to_restore["best_acc"]

	#cudnn.benchmark = True
	eval('setattr(torch.backends.cudnn, "benchmark", True)')
	
	if args.cuda_deterministic:
		logger.info("cuda deterministic")
		eval('setattr(torch.backends.cudnn, "deterministic", True)') 

	# if 'save' in args.mode:
	# 	def _save(loader, prefix, args):
	# 		targets, rep, correct, pred, logits, top1acc, top5acc =  save_feature(loader,model, classifier)
	# 		if 'acc' in args.mode:
	# 			np.array([top1acc]).dump(os.path.join(args.dump_path, f'{prefix}_acc_{args.rank}.npy'))
				
	# 			print(top1acc)
	# 		if 'rep' in args.mode:
	# 			rep = rep.astype(np.float16)
	# 			rep.dump(os.path.join(args.dump_path, f'{prefix}_represent_{args.rank}.npy'))
				
	# 		if 'pred' in args.mode : 
	# 			pred.dump(os.path.join(args.dump_path, f'{prefix}_pred_{args.rank}.npy'))
	# 			correct.dump(os.path.join(args.dump_path, f'{prefix}_corrects_{args.rank}.npy'))
	# 		if 'prob' in args.mode:
	# 			logits.dump(os.path.join(args.dump_path, f'{prefix}_logits_{args.rank}.npy'))

	# 			prob = np.exp(logits) / np.exp(logits).sum(axis=1,keepdims=True) 
	# 			prob = prob.max(axis=1)
	# 			prob.dump(os.path.join(args.dump_path, f'{prefix}_prob_{args.rank}.npy'))
				
	# 			# multi-class version adaboost called samme.r https://hastie.su.domains/Papers/samme.pdf (page 9)
	# 			weight = -((datamsg['nclass']-1) / datamsg['nclass']) * np.log(prob + 1e-8) * (pred == targets)
	# 			weight.dump(os.path.join(args.dump_path, f'{prefix}_weight_{args.rank}.npy'))
	# 			h = (datamsg['nclass']-1) * (np.log(prob + 1e-8))

	# 		targets.dump(os.path.join(args.dump_path, f'{prefix}_targets_{args.rank}.npy'))

	# 	for key in additional_loaders:
	# 		if key in args.mode:
	# 			_save(additional_loaders[key], key, args)
	# 	if '_train_' in args.mode:
	# 		_save(train_loader, 'train', args)
	# 	if '_val_' in args.mode:
	# 		_save(val_loader, 'val', args)

	# 	return 
	
	# if args.mode == 'eval_only':
	# 	indices = datamsg['targetmask'] if 'targetmask' in datamsg else None
	# 	loss, top1, top5 = validate_network(val_loader, model, classifier,args, indices)
	# 	logger.info(
	# 				"Test:\t"
	# 				"Loss {loss:.4f}\t"
	# 				"Acc@1 {top1:.3f}\t".format(loss=loss, top1=top1))
	# 	return 
	
	# if args.mode == 'eval_train_only':
	# 	indices = datamsg['targetmask'] if 'targetmask' in datamsg else None
	# 	loss, top1, top5 = validate_network(train_loader, model, classifier,args, indices)
	# 	logger.info(
	# 				"Train:\t"
	# 				"Loss {loss:.4f}\t"
	# 				"Acc@1 {top1:.3f}\t".format(loss=loss, top1=top1))
	# 	return 


	for epoch in range(start_epoch, args.epochs):
		
		if epoch == 0 and args.save_init:
			save_dict = {
					"epoch": 0,
					"state_dict": classifier.state_dict(),
					"optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict(),
					"best_acc": 0,
				}
			torch.save(save_dict, os.path.join(args.dump_path, f"checkpoint_init.pth.tar"))
			logger.info('saved weight initialization')
		# train the network for one epoch
		logger.info("============ Starting epoch %i ... ============" % epoch)

		# set samplers
		train_loader.sampler.set_epoch(epoch)

		tr_epoch, tr_loss, tr_top1, tr_top5 = train(model, classifier, optimizer, train_loader, epoch, args)
		scheduler.step()

		if (epoch+1) % args.eval_freq == 0: 
			loss, top1, top5 = validate_network(val_loader, model,  classifier, args,)
			

			if args.custom_eval_func is not None: 
				from src import custom_eval
				for custom_eval_name in args.custom_eval_func:
					custom_eval_func = getattr(custom_eval, custom_eval_name)
					custom_eval_results = custom_eval_func(val_loader, model, classifier, args)
					logger.info(f'{custom_eval_name}: ' + ','.join(['%.4f' % val for val in custom_eval_results]))
			
			# additional validation sets
			additional_msg = {}
			for key in additional_loaders:
				loader = additional_loaders[key]
				ad_loss, ad_top1, ad_top5 = validate_network(loader, model,  classifier,args)
				additional_msg[key]=[ad_loss, ad_top1, ad_top5]

			training_stats.update([tr_epoch, tr_loss, tr_top1, tr_top5] + [loss, top1, top5])


			# log best acc
			#global best_acc
			is_best = False 
			if top1 > best_acc:
				#best_acc = top1.avg.item()
				best_acc = top1
				is_best = True 

			if args.rank == 0:
				logger.info(
					"Test:\t"
					"Loss {loss:.4f}\t"
					"Acc@1 {top1:.3f}\t"
					"Best Acc@1 so far {acc:.1f}".format(loss=loss, top1=top1, acc=best_acc))
				
				for key in additional_msg:
					loss, top1, _ = additional_msg[key]
					logger.info(
						"additional Test {key}:\t"
						"Loss {loss:.4f}\t"
						"Acc@1 {top1:.3f}".format(key=key, loss=loss, top1=top1))

			# save checkpoint
			if args.rank == 0:

				save_dict = {
					"epoch": epoch + 1,
					"state_dict": classifier.state_dict(),
					"optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict(),
					"best_acc": best_acc,
				}
				torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))
				
				if (epoch+1) % args.save_freq == 0:
					torch.save(save_dict, os.path.join(args.dump_path, f"checkpoint_epoch{epoch+1}.pth.tar"))
		
				if is_best:
					torch.save(save_dict, os.path.join(args.dump_path, "best.pth.tar"))
			
			
	logger.info("Training of the supervised linear classifier on frozen features completed.\n"
				"Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, reglog, optimizer, loader,  epoch, args ):
	"""
	Train the models on the dataset.
	"""
	# running statistics
	batch_time = AverageMeter()
	data_time = AverageMeter()

	# training statistics
	top1 = AverageMeter()
	top5 = AverageMeter()
	losses = AverageMeter()
	end = time.perf_counter()
	
	model.eval()
	reglog.train()

	criterion = nn.CrossEntropyLoss().cuda()

	for iter_epoch, record in enumerate(loader):
		# measure data loading time
		data_time.update(time.perf_counter() - end)
		

		if len(record) == 2:
			inp, target = record 
		elif len(record) == 3:
		
			inp, target, meta = record 
		
		#move to gpu
		inp = inp.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)
		# forward

		with torch.no_grad():
			output = model(inp)

		output = reglog(output)
		loss = criterion(output, target) 

		# compute the gradients
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# update stats
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), inp.size(0))
		top1.update(acc1[0], inp.size(0))
		top5.update(acc5[0], inp.size(0))

		batch_time.update(time.perf_counter() - end)
		end = time.perf_counter()

		# verbose
		if args.rank == 0 and iter_epoch % 50 == 0:
			
			logger.info(
				"Epoch[{0}] - Iter: [{1}/{2}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Prec {top1.val:.3f} ({top1.avg:.3f})\t"
				"LR {lr}".format(
					epoch,
					iter_epoch,
					len(loader),
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					top1=top1,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

	return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, classifier, args, indices=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	#global best_acc

	# switch to evaluate mode
	model.eval()
	classifier.eval()

	criterion = nn.CrossEntropyLoss().cuda()

	with torch.no_grad():
		end = time.perf_counter()
		for i, record in enumerate(val_loader):
			if len(record) == 2:
				inp, target = record 
			elif len(record) == 3:
				inp, target, meta = record 
			
			# move to gpu
			inp = inp.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			# compute output
			output = classifier(model(inp))
			
	


			if indices is not None:
				output = output[:,indices]

			loss = criterion(output, target)
	
			acc1, acc5 = accuracy(output, target, topk=(1, 5))

			losses.update(loss.item(), inp.size(0))

			top1.update(acc1[0], inp.size(0))
			top5.update(acc5[0], inp.size(0))

			# measure elapsed time
			batch_time.update(time.perf_counter() - end)
			end = time.perf_counter()
			if args.rank == 0 and i % 50 == 0:
				logger.info(
				"Epoch[{0}] - Iter: [{1}/{2}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Prec {top1.val:.3f} ({top1.avg:.3f})\t".format(
					0,
					i,
					len(val_loader),
					batch_time=batch_time,
					loss=losses,
					top1=top1,
				)
				)

	scores_val = torch.Tensor(np.array([losses.sum, top1.sum.item(), top5.sum.item(), \
								losses.count, top1.count, top5.count])).to(target.get_device())
	dist.all_reduce(scores_val, op=dist.ReduceOp.SUM)
	scores_val = tuple((scores_val[:3] / scores_val[3:]).detach().cpu().numpy().tolist())
	losses, top1, top5  = scores_val
	return losses, top1, top5 

# def save_feature(loader, model,reglog): 
	
# 	top1 = AverageMeter()
# 	top5 = AverageMeter()
# 	model.eval()
# 	if reglog is not None:
# 		reglog.eval() 
   
# 	rep = [] 
# 	corrects = [] 
# 	pred = [] 
# 	alllogits = []
# 	targets = [] 

# 	with torch.no_grad():
# 		for i, record in enumerate(loader):
# 			if len(record) == 2:
# 				inp, target = record 
# 			elif len(record) == 3:
# 				inp, target, meta = record 

# 			inp = inp.cuda(non_blocking=True)
# 			target = target.cuda(non_blocking=True)
# 			represent = model(inp)
# 			#print(represent.shape)
# 			if reglog is not None:
# 				logits = reglog(represent)
# 				correct = logits.argmax(axis=1) == target 
# 				corrects.append(correct.detach().cpu().numpy().flatten())
# 				pred.append(logits.argmax(axis=1).detach().cpu().numpy().flatten())
# 				alllogits.append(logits.detach().cpu().numpy())
# 				acc1, acc5 = accuracy(logits, target, topk=(1, 5))
				
# 				top1.update(acc1[0], inp.size(0))
# 				top5.update(acc5[0], inp.size(0))

# 				if  i % 50 == 0:
# 					print(
# 					"Epoch[{0}] - Iter: [{1}/{2}]\t"
# 					"Prec {top1.val:.3f} ({top1.avg:.3f})\t".format(
# 						0,
# 						i,
# 						len(loader),
# 						top1=top1,
# 					)
# 					)
			
# 			targets.append(target.detach().cpu().numpy().flatten())
# 			rep.append(represent.detach().cpu().numpy())
			

# 	rep = np.concatenate(rep, axis=0)
# 	targets = np.concatenate(targets, axis=0)

	
# 	if reglog is not None:
# 		pred = np.concatenate(pred, axis=0)
# 		alllogits = np.concatenate(alllogits, axis=0)
# 		corrects = np.concatenate(corrects)
# 		return targets, rep, corrects, pred, alllogits, top1.avg.item(), top5.avg.item()
# 	return targets, rep, None, None, None, 0, 0

def custom_params():
	parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

	#########################
	#### main parameters ####
	#########################
	parser.add_argument("--config", type=str, default=None,
						help="yaml config path")
	
	parser.add_argument("--dump_path", type=str, default=".",
						help="experiment dump path for checkpoints and log")
	parser.add_argument("--seed", type=int, default=31, help="seed")
	parser.add_argument("--data_path", type=str, default="data/imagenet",
						help="path to dataset repository")
	parser.add_argument("--data_name", type=str, default="imagenet1k",
						help="name of datasets [imagenet1k,]")
	parser.add_argument("--tf_name", type=str, default="eval",
						help="name of datasets transform [eval,224px,284px]")
	parser.add_argument("--workers", default=8, type=int,
						help="number of data loading workers")
	parser.add_argument("--data_rate", default=1, type=float,
						help="rate of data to use")

	parser.add_argument("--reweight_path", default=None, type=str, help='path to reweight file')	
	

	parser.add_argument("--custom_eval_func", nargs='*', default=None, type=str, help='custom evaluation function')

	#########################
	#### model parameters ###
	#########################
	
	parser.add_argument("--tag", default=None, type=str, help='a tag help load --arch, --pretrained parameters')
	
	parser.add_argument("--arch",	   default="resnet50", nargs='*', type=str, help="convnet architecture")
	parser.add_argument("--pretrained", default="",		 nargs='*', type=str, help="path to pretrained weights")
	

	#parser.add_argument("--fix1st_pretrained", default="", type=str, help="path to pretrained weights (1st layer). only for *fix1st model")
	#parser.add_argument('--dist_clf', default=False, type=bool_flag,help='use cosine classifier or not ')
	parser.add_argument('--skip_pool', default=False, type=bool_flag,help='skip pool or not ')
	

	parser.add_argument("--use_bn", default=False, type=bool_flag, help="optionally add a batchnorm layer before the linear classifier")
	parser.add_argument('--classifier', default='linear', type=str, help='classifier  [linear, convpoollinear_k]')
	parser.add_argument('--richway', default='cat', type=str, help='[cat]')
	
	parser.add_argument('--headinit', default='none', type=str, help='init head: none, dumped_weights, cat_weights, normal')
	parser.add_argument('--headpretrained', default=None,  nargs='*',type=str, help='path to dumped head weights. It is activate when --headcatinit is dumped_weight')
	
	parser.add_argument('--exp_mode', default='lineareval', type=str, help='lineareval, finetune')
	parser.add_argument('--mode', default='train', type=str, help='train, eval_only, save_val_prob')
	
	#parser.add_argument("--reinit_head", default=True, type=bool_flag, help="")
	parser.add_argument('--sync_bn', default=False, type=bool_flag,help='sync_bn')
	
 
	#########################
	#### optim parameters ###
	########jvihnrbutvvthiguleudcchcjrknunbc#################
	parser.add_argument("--optimizer", default='sgd', type=str, help='sgd, adam')
	parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
	parser.add_argument("--wd_skip_bn", default=False, type=bool_flag, help="")
	#parser.add_argument("--nesterov", default=True, type=bool_flag, help="nesterov momentum")
	parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum") #March 7th, 2023. change the default value to False 
	parser.add_argument("--momentum", default=0.9, type=float, help="momentum in SGD, beta1 in adam")
	parser.add_argument("--beta2", default=0.99, type=float, help="beta2 in adam")
	
	parser.add_argument("--epochs", default=28, type=int,
						help="number of total epochs to run")
	parser.add_argument("--batch_size", default=32, type=int,
						help="batch size per gpu, i.e. how many unique instances per gpu")
	parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
	parser.add_argument("--lr_last_layer", default=None, type=float, help="initial learning rate")
	
	parser.add_argument("--scheduler_type", default="step", type=str, choices=["step", "cosine"])
	# for multi-step learning rate decay
	parser.add_argument("--decay_epochs", type=int, nargs="+", default=[8, 16, 24],
						help="Epochs at which to decay learning rate.")
	parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
	# for cosine learning rate schedule
	parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
	
	parser.add_argument("--eval_freq", default=1, type=int, help="frequency to do evaluation")
	parser.add_argument("--save_freq", default=999, type=int, help="frequency to save checkpoint_epochi.pth.tar")
	parser.add_argument("--save_init", default=False, type=bool_flag, help="save weights initialization")
	


	#########################
	#### dist parameters ###
	#########################
	parser.add_argument("--dist_url", default="env://", type=str,
						help="url used to set up distributed training")
	parser.add_argument("--world_size", default=-1, type=int, help="""
						number of processes: it is set automatically and
						should not be passed as argument""")
	parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
						it is set automatically and should not be passed as argument""")
	# parser.add_argument("--local_rank", default=0, type=int,
	#					 help="this argument is not used and should be ignored")
	parser.add_argument('--debug',action='store_true', help='debug mode')
	parser.add_argument('--gpu', default=None, type=int, help='gpu to use')
	
	parser.add_argument('--cuda_deterministic', action='store_true',help='cuda deterministic. slow but deterministic')
	


	#########################
	#### slurm parameters ###
	#########################
	parser = add_slurm_params(parser)
	args = parser.parse_args()

	return args 


if __name__ == "__main__":
	
	args = custom_params()
	main(args)

	