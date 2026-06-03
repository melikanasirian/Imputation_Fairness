import numpy as np
import torch
from geomloss import SamplesLoss
from utils import nanmean, MAE, RMSE
import logging


class SinkhornImputation():
    def __init__(self, 
                 eps=0.01, 
                 lr=1e-2, 
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
        torch.manual_seed(42)
        np.random.seed(42)

        X = X.clone()
        n, d = X.shape
        
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger than half size = {n // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape, dtype=X.dtype).to(X.device) + nanmean(X, 0))[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps

            sk_loss = 0
            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)
                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                sk_loss += self.sk(X1, X2)

            if torch.isnan(sk_loss).any() or torch.isinf(sk_loss).any():
                logging.info("NaN or Inf detected in Sinkhorn loss. Stopping training.")
                break

            optimizer.zero_grad()
            sk_loss.backward(retain_graph=True)

            if imps.grad is not None:
                print(f"[GRAD] Sinkhorn grad norm: {imps.grad.norm():.6f}")
                imps.grad.zero_()

            optimizer.step()

            if X_true is not None:
                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()

            if verbose and (i % report_interval == 0):
                log_msg = f'Iteration {i}:\t Loss: {sk_loss.item() / self.n_pairs:.4f}'
                if X_true is not None:
                    log_msg += f'\t MAE: {maes[i]:.4f}\t RMSE: {rmses[i]:.4f}'
                logging.info(log_msg)

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled