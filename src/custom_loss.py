import torch
import torch.nn as nn
import torch.autograd as autograd
from wilds.common.utils import split_into_groups
import numpy as np 
import torch.nn.functional as F
class BalanceGrad():
    def __init__(self, drop_f, drop_b, num_feats, num_classes, device):
        self.drop_f = (1-drop_f)*100
        self.drop_b = (1-drop_b)*100
        self.num_classes = num_classes
        self.device = device
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.prob
    def update(self, x, featurizer, classifier,y):
        all_y = y 
        feat = featurizer(x)
        logits = classifier(feat)
        loss = self.criterion(logits, y)
        grads = autograd.grad(loss, feat)[0]

        


        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = featurizer(x)
        # predictions
        all_p = classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(self.device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        #print(mask)
        all_p_muted_again = classifier(all_f * mask)
        #print(mask)
        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)

        return loss, all_p
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return {'loss': loss.item()}
class Dynamicdropout():
    def __init__(self, drop_f, drop_b, num_classes, device):
        self.drop_f = (1-drop_f)*100
        self.drop_b = (1-drop_b)*100
        self.num_classes = num_classes
        self.device = device
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.dropout = torch.zeros(1024).cuda()
    def update(self, x, featurizer, classifier,y):
        

        all_y = y 
        # features
        all_f = featurizer(x)
        # predictions
        all_p = classifier(all_f)
        loss = F.cross_entropy(all_p, all_y)
        all_g = autograd.grad(loss, all_f)[0].sum(axis=0)
        
        percentiles = np.percentile(all_g.cpu(), self.drop_f)
        
        index = all_g.lt(percentiles).detach().float()

        self.dropout += index
        #self.dropout /= self.dropout.max() 

        print(self.dropout[:512].mean(), self.dropout[512:].mean())

        #output = classifier(all_f * self.dropout  )
        #loss = self.criterion(output, all_y)

        output = classifier(all_f)
        loss = self.criterion(output, all_y)
        

        return loss, all_p
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return {'loss': loss.item()}
class RSC():
    def __init__(self, drop_f, drop_b, num_classes, device):
        self.drop_f = (1-drop_f)*100
        self.drop_b = (1-drop_b)*100
        self.num_classes = num_classes
        self.device = device
    def update(self, x, featurizer, classifier,y):
        # device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # # inputs
        # all_x = torch.cat([x for x, y in minibatches])
        # # labels
        # all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_y = y 
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = featurizer(x)
        # predictions
        all_p = classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]
        
        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(self.device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        #print(mask)
        all_p_muted_again = classifier(all_f * mask)
        #print(mask)
        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)

        return loss, all_p
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return {'loss': loss.item()}


class SD():
	def __init__(self,sd_lambda, device,is_training=True  ):
		self.sd_lambda = sd_lambda
		self.device = device
		self.is_training = is_training
		self.loss = nn.CrossEntropyLoss().to(self.device)
	def update(self,pred,y,g=None):
		#assert g is None 
		loss = self.loss(pred, y)
		penalty = (pred ** 2).mean()

		return loss + self.sd_lambda * penalty

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
