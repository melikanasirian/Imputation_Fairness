import numpy as np
import torch
from geomloss import SamplesLoss
from utils import nanmean, MAE, RMSE
from CMI_torch import (
    estimate_CMI_soft_kronecker_gaussian,
    estimate_CMI_gumbel_softmax_kernel,
    estimate_CMI_separate_kernel, generate_distributions_for_discrete_data,
)
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


class SinkhornImputation_CMI():
    def __init__(self,
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=2000,
                 highest_lamda_cmi=100,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9,
                 lambda_cmi=0.1,
                 cmi_index=0):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        self.lambda_cmi = lambda_cmi
        self.highest_lamda_cmi = highest_lamda_cmi
        self.cmi_index = cmi_index

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None,
                      X_cols=None, Y_cols=None, Z_cols=None,
                      encoder=None, discrete_columns=None, continuous_columns=None,
                      Y=None, data_processed=None, one_hot_encoded=None, continuous_part=None):

        torch.manual_seed(42)
        np.random.seed(42)

        X = X.clone()
        n, d = X.shape

        mask = torch.isnan(X).double()

        imps = (self.noise * torch.randn(mask.shape, dtype=X.dtype).to(X.device) + nanmean(X, 0))[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

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

            self.lambda_cmi = min(self.highest_lamda_cmi, i / 100.0)

            assert len(X_cols) == len(Y_cols) == len(Z_cols)
            cmi = None

            for j in range(len(X_cols)):
                triplet = (X_cols[j], Y_cols[j], Z_cols[j])

                if self.cmi_index == 0:
                    cmi_input = X_filled
                    tcmi = estimate_CMI_soft_kronecker_gaussian(
                        cmi_input, triplet, continuous_columns, discrete_columns
                    )

                elif self.cmi_index == 1 or self.cmi_index == 2:
                    if Y is None:
                        raise ValueError("Y must be provided when using CMI Method 2 or 3.")
                    data_continuous = X_filled[:, continuous_columns]
                    discrete_distributions = generate_distributions_for_discrete_data(X_filled, discrete_columns, encoder)
                    data_combined_tensor = torch.cat([discrete_distributions, data_continuous], dim=1)

                    if self.cmi_index == 1:
                        tcmi = estimate_CMI_gumbel_softmax_kernel(
                            data_combined_tensor, triplet, discrete_columns, continuous_columns, encoder,
                        )
                    else:
                        tcmi = estimate_CMI_separate_kernel(
                            data_combined_tensor, triplet, discrete_columns, continuous_columns, encoder
                        )

                cmi = tcmi if cmi is None else cmi + tcmi

            cmi /= len(X_cols)
            total_loss = sk_loss + self.lambda_cmi * cmi

            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            total_loss.backward()
            # print(f"Sinkhorn loss: {sk_loss.item()}, CMI loss: {cmi.item()}, total loss: {total_loss.item()}")
            optimizer.step()

            if X_true is not None:
                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()

            if verbose and (i % report_interval == 0):
                print(f"Iteration {i}: Sinkhorn={sk_loss.item():.4f}, CMI={cmi.item():.4f}")

        if X_true is not None:
            return X_filled, maes, rmses, {}
        else:
            return X_filled, {}
