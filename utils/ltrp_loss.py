import torch

import torch.nn.functional as F
import torch.nn as nn

def kld(p, q):
    # p : batch x n_items
    # q : batch x n_items
    return (p * torch.log2(p / q)).sum()


def jsd(predict, target):
    # predict : batch x n_items
    # target : batch x n_items
    top1_true = F.softmax(target, dim=0)
    top1_pred = F.softmax(predict, dim=0)
    jsd = 0.5 * kld(top1_true, top1_pred) + 0.5 * kld(top1_pred, top1_true)
    return jsd


class list_mle(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        # y_pred : batch x n_items
        # y_true : batch x n_items
        if self.k is not None and self.k > 0:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=[self.k])).long()
            y_pred = y_pred[:, sublist_indices]
            y_true = y_true[:, sublist_indices]

        _, indices = y_true.sort(descending=True, dim=-1)

        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)

        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true

        return listmle_loss.sum(dim=1).mean()


class list_mleEx(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        # y_pred : batch x n_items
        # y_true : batch x n_items
        if self.k is not None and self.k > 0:
            N, L = y_true.shape
            noise = torch.rand(N, L, device=y_true.device)
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_keep = ids_shuffle[:, :self.k]
            y_pred = torch.gather(y_pred, dim=1, index=ids_keep)
            y_true = torch.gather(y_true, dim=1, index=ids_keep)

        _, indices = y_true.sort(descending=True, dim=-1)

        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)

        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true

        return listmle_loss.sum(dim=1).mean()



class rank_net(nn.Module):
    def __init__(self, thread=0.001, sigma=1):
        super().__init__()
        self.thread = thread
        self.sigma = sigma

    def get_target_prob(self, matrix):
        #temp = y_true.unsqueeze(-1) - y_true.unsqueeze(-2)
        values = torch.zeros_like(matrix)  # zeros
        matrix = torch.where(torch.gt(matrix, -self.thread) & torch.lt(matrix, self.thread), values, matrix)
        values.add_(1)  # ones
        matrix = torch.where(matrix >= self.thread, values, matrix)
        values.add_(-2)  # minus ones
        matrix = torch.where(matrix <= -self.thread, values, matrix)
        ret = 0.5 * (1 + matrix)
        return ret

    def get_pred_prob(self, matrix):
        #temp = y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2)
        ret = 1 / (1 + torch.exp(-self.sigma * matrix))
        return ret

    def forward(self, y_pred, y_true):
        matrix = y_true.unsqueeze(-1) - y_true.unsqueeze(-2)
        target_prob = self.get_target_prob(matrix)
        matrix = y_pred.unsqueeze(-1) - y_pred.opunsqueeze(-2)
        pred_prob = self.get_pred_prob(matrix)
        loss = - target_prob * torch.log(pred_prob) - (1 - target_prob) * torch.log(1 - pred_prob)
        loss = loss.mean() / 2
        return loss

    def forward(self, y_pred, y_pred_partial, y_true, y_true_partial):
        matrix = y_true.unsqueeze(-1) - y_true_partial.unsqueeze(-2)
        target_prob = self.get_target_prob(matrix)
        matrix = y_pred.unsqueeze(-1) - y_pred_partial.unsqueeze(-2)
        pred_prob = self.get_pred_prob(matrix)
        loss = - target_prob * torch.log(pred_prob) - (1 - target_prob) * torch.log(1 - pred_prob)
        loss = loss.mean() / 2
        return loss

class list_net(nn.Module):
    def __init__(self, t=0.5):
        super().__init__()
        self.t = t

    def forward(self, y_pred, y_true):
        y_prob = F.softmax(y_true, dim=-1)
        loss = - y_prob * F.log_softmax(y_pred, dim=-1)
        loss = loss.mean()
        return loss

class point_wise(nn.Module):
    def __init__(self):
        super(point_wise, self).__init__()

    def forward(self,  y_pred, y_true):
        return F.mse_loss(input=y_pred, target=y_true)


class focused_rank(nn.Module):
    def __init__(self, k=20 ):
        super(focused_rank, self).__init__()
        self.k = k
        self.list_wise = list_mle()
        self.pair_wise = rank_net()

    def forward(self, y_pred, y_true):
        _, sorted_ids = torch.sort(y_true, dim=-1, descending=True)
        ids_topk = sorted_ids[:, :self.k]
        ids_other = sorted_ids[:, self.k:]

        y_pred_topk = torch.gather(y_pred, dim=1, index=ids_topk)
        y_true_topk = torch.gather(y_true, dim=1, index=ids_topk)
        y_pred_other = torch.gather(y_pred, dim=1, index=ids_other)
        y_true_other = torch.gather(y_true, dim=1, index=ids_other)

        loss_list_wise = self.list_wise(y_pred_topk, y_true_topk)
        loss_pair_wise = self.pair_wise(y_pred_topk, y_pred_other, y_true_topk, y_true_other)

        return loss_list_wise + loss_pair_wise



x = torch.arange(0, 25)
print(x)
x = x.unsqueeze(0).repeat(256, 1)
loss = list_mleEx(25)

print(loss(25 - x, x))