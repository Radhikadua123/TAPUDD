import torch
import torch.nn as nn
import math



class SVD_L(torch.nn.Module):
    def __init__(self):
        super(SVD_L, self).__init__()

    def forward(self, in_feature):
        in_feature = in_feature.to(torch.float32)
        loss = torch.tensor(1.).to('cuda')/(torch.mean(torch.var(in_feature, dim=0)) + (1e-6))
        return loss

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, in_feature):
        corr = self.corr(in_feature.T) ** 2

        loss = torch.mean(corr[torch.tril(torch.ones_like(corr), diagonal=-1) == 1])
        # cov = self.cov(x.T) + 1e-6*torch.eye(x.size(1)).to('cuda')
        # # cov = self.cov(x.T)
        # # loss = -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim=0), covariance_matrix=cov).entropy()
        # loss = -torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim=0),
        #                                                                    scale_tril=torch.tril(cov)).entropy()

        return loss

    def cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)
    def corr(self, x):
        # calculate covariance matrix of rows
        c = self.cov(x)

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c) + 1e-8)
        c = c.div(stddev.expand_as(c).t() + 1e-8)

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        # c = torch.clamp(c, -1.0, 1.0)

        return c

class MULoss(nn.Module):
    def __init__(self, args):
        super(MULoss, self).__init__()
        self.num_class = args.num_classes

    def forward(self, in_feature, label):
        in_feature = in_feature.to(torch.float32)
        mu_matrix = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)
        # pairwise_distance = torch.cdist(class_mu, class_mu, p=2)
        # distance = pairwise_distance[
        #     torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        # term2 = -distance
        # loss_mu = torch.mean(term2)

        pairwise_distance = torch.cdist(mu_matrix, mu_matrix, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular
        loss_mu = - torch.mean(distance) / math.sqrt(in_feature.size(1))

        return loss_mu