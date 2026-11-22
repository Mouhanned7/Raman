"""Regression models: PLS (chemometrics standard), Ridge, and a 1D CNN."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def fit_pls(Xtr, ytr, max_comp=20, seed=0):
    """PLS with the number of latent variables chosen by 5-fold CV on the training set."""
    best, best_rmse = 1, np.inf
    kf = KFold(5, shuffle=True, random_state=seed)
    for k in range(2, max_comp + 1, 2):
        pred = cross_val_predict(PLSRegression(k, scale=False), Xtr, ytr, cv=kf).ravel()
        rmse = np.sqrt(np.mean((pred - ytr) ** 2))
        if rmse < best_rmse:
            best, best_rmse = k, rmse
    model = PLSRegression(best, scale=False).fit(Xtr, ytr)
    model.n_components_chosen = best
    return model


def fit_ridge(Xtr, ytr, alpha=100.0):
    sc = StandardScaler().fit(Xtr)
    model = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return lambda X: model.predict(sc.transform(X))


class SpectrumCNN(nn.Module):
    def __init__(self, n_points):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.conv = nn.Sequential(block(1, 16, 9), block(16, 32, 7), block(32, 64, 5))
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, n_points)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(n_flat, 64),
                                  nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(1)


def fit_cnn(Xtr, ytr, epochs=60, lr=2e-3, batch=64, seed=0, device="cpu"):
    """Trains the CNN; returns a predict function (input spectra -> concentration)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-9)
    scaler = StandardScaler().fit(Xtr)          # area-normalised spectra have tiny values
    X = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    y = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    model = SpectrumCNN(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(X) + batch - 1) // batch))
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb = X[idx]
            xb = xb * (1 + 0.05 * torch.randn(len(idx), 1)) + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
            sched.step()

    def predict(Xnew):
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(scaler.transform(Xnew), dtype=torch.float32).to(device)).cpu().numpy()
        return out * sd + mu
    return predict
