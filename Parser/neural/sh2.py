import torch
from torch import Tensor, logsumexp
from torch.autograd import Function


class Sinkhorn(Function):
    @staticmethod
    def forward(ctx, costs: Tensor, left: Tensor, right: Tensor, num_iter: int, scaling_factor: float):
        costs /= scaling_factor
        for _ in range(num_iter):
            costs -= logsumexp(costs, dim=1, keepdim=True)
            costs -= logsumexp(costs, dim=2, keepdim=True)
        ctx.save_for_backward(costs, )
        return costs
        # log_p = -costs / scaling_factor
        # log_a = torch.log(left).unsqueeze(dim=1)
        # log_b = torch.log(right).unsqueeze(dim=0)
        # for _ in range(num_iter):
        #     log_p -= (torch.logsumexp(log_p, dim=0, keepdim=True) - log_b)
        #     log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_a)
        # p = torch.exp(log_p)
        # ctx.save_for_backward(p, torch.sum(p, dim=1), torch.sum(p, dim=0))
        # ctx.lambd_sink = scaling_factor
        # return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors
        m, n = p.shape
        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat((torch.cat((torch.diag(a), p), dim=1),
                       torch.cat((p.T, torch.diag(b)), dim=1)),
                      dim=0)[:-1, :-1]
        t = torch.cat((
            grad_p.sum(dim=1),
            grad_p[:, :-1].sum(dim=0)),
            dim=0).unsqueeze(1)
        grad_ab, _ = torch.solve(t, K)
        grad_a = grad_ab[:m, :]
        grad_b = torch.cat((grad_ab[m:, :], torch.zeros([1, 1], device='cuda', dtype=torch.float32)), dim=0)
        U = grad_a + grad_b.T
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=1)
        return grad_p, grad_a, grad_b, None, None
