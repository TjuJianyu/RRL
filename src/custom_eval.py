from src.utils import AverageMeter, accuracy
import torch 
import torch.nn as nn 
import torch.distributed as dist
import numpy as np 
def feature_importance(val_loader, model,  classifier, args):
	try:
		weight = classifier.module.classifier.linear.weight

		feat_im = weight.abs().mean(axis=0)
		assert len(args.arch) == 1 

		if len(args.arch) > 1:
			n_blocks = len(args.arch)
		else:
			n_blocks = ''
			for var in args.arch[0]:
				if var not in [str(num) for num in range(10)]:
					break 
				n_blocks += var 
			n_blocks = int(n_blocks) if len(n_blocks) > 0 else 1

		feat_im = feat_im.reshape(n_blocks, len(feat_im) // n_blocks).mean(axis=1).flatten()

		return feat_im

	except Exception as e:
		print(e)
		return None 

def cat_validate_network(val_loader, model, classifier, args, indices=None):
	

	

	# switch to evaluate mode
	model.eval()
	classifier.eval()
	
	if len(args.arch) > 1:
			n_blocks = len(args.arch)
	else:
		n_blocks = ''
		for var in args.arch[0]:
			if var not in [str(num) for num in range(10)]:
				break 
			n_blocks += var 
		n_blocks = int(n_blocks) if len(n_blocks) > 0 else 1



	losses = [AverageMeter() for i in range(n_blocks)]
	top1 = [AverageMeter() for i in range(n_blocks)]
	top5 = [AverageMeter() for i in range(n_blocks)]


	criterion = nn.CrossEntropyLoss().cuda()
	weight = classifier.module.classifier.linear.weight.T
	bias = classifier.module.classifier.linear.bias 
	classifierlinear = classifier.module.classifier.linear
	classifier.module.classifier.linear = nn.Identity() 
	with torch.no_grad():
		
		for i, record in enumerate(val_loader):
			if len(record) == 2:
				inp, target = record 
			elif len(record) == 3:
				inp, target, meta = record 
			
			# move to gpu
			inp = inp.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)
			

			# compute output
			rep = classifier(model(inp))
			feat_dim = rep.shape[1] // n_blocks
			for block in range(n_blocks):
				output = rep[:,feat_dim * block:feat_dim * (block+1)] @ weight[feat_dim * block:feat_dim * (block+1),:] + bias 

				if indices is not None:
					output = output[:,indices]

				loss = criterion(output, target)
				#print(indices,output)
				acc1, acc5 = accuracy(output, target, topk=(1, 5))
				#print(acc1, )
				losses[block].update(loss.item(), inp.size(0))
				#losses.update(0, inp.size(0))
				top1[block].update(acc1[0], inp.size(0))
				top5[block].update(acc5[0], inp.size(0))

	classifier.module.classifier.linear = classifierlinear
			
	scores_val = torch.Tensor(np.array([[losses[block].sum, top1[block].sum.item(), top5[block].sum.item(), \
								losses[block].count, top1[block].count, top5[block].count] for block in range(n_blocks)])).to(target.get_device())
	dist.all_reduce(scores_val, op=dist.ReduceOp.SUM)

	scores_val =  (scores_val[:,:3] / scores_val[:,3:]).detach().cpu().numpy().flatten().tolist()

	return scores_val