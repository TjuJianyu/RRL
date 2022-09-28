import torch
import torch.nn as nn
import torch.autograd as autograd
from wilds.common.utils import split_into_groups
class IRM():
	"""docstring for IRM"""
	def __init__(self, irm_lambda, device,  is_training=True ):
		self.irm_lambda = irm_lambda 
		self.device = device
		self.scale = torch.tensor(1.).to(self.device).requires_grad_()
		self.loss = nn.CrossEntropyLoss(reduction='none').to(self.device)
		self.is_training=is_training
		#self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
		#self.optimizer = optimizer


	def irm_penalty(self, losses):
		grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
		grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
		result = torch.sum(grad_1 * grad_2)
		return result

	def objective(self, pred, y, g):

		
		unique_groups, group_indices, _ = split_into_groups(g)
		n_groups_per_batch = unique_groups.numel()
		avg_loss = 0.
		penalty = 0.


		for i_group in group_indices: # Each element of group_indices is a list of indices
			group_losses = self.loss(self.scale * pred[i_group], y[i_group])

			if group_losses.numel()>0:
				avg_loss += group_losses.mean()
			if self.is_training: # Penalties only make sense when training
				penalty += self.irm_penalty(group_losses)

		avg_loss /= n_groups_per_batch
		penalty /= n_groups_per_batch

		# if self.update_count >= self.irm_penalty_anneal_iters:
		# 	penalty_weight = self.irm_lambda
		# else:
		# 	penalty_weight = 0
		penalty_weight =self.irm_lambda	

		loss = avg_loss + penalty * penalty_weight
		
		if penalty_weight > 1:
		   loss /= penalty_weight
	
		return loss 

	def update(self, pred, y, g):
		# if self.update_count == self.irm_penalty_anneal_iters:
			# print('Hit IRM penalty anneal iters')
			# # Reset optimizer to deal with the changing penalty weight
			# self.optimizer = initialize_optimizer(self.config, self.model)

		#self.update_count += 1
		#print(g)
		return self.objective(pred,y,g)
