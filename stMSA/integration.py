import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_conv import GATConv


class stIntegration(nn.Module):

    def __init__(self, dims, centroids, batchs):

        super().__init__()

        # encoder
        self.encoder_0 = GATConv(dims[0], dims[1])
        self.encoder_1 = nn.Parameter(torch.Tensor(dims[1], dims[2]))

        # decoder
        self.decoder_0 = GATConv(dims[2], dims[1])
        self.decoder_1 = nn.Parameter(torch.Tensor(dims[1], dims[0]))

        # # DEC init centroids
        self.centroids = centroids
        self.batchs = batchs

        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_normal_(self.encoder_1.data, gain=1.414)
        nn.init.xavier_normal_(self.decoder_1.data, gain=1.414)


    def target_distribution(self, q_list):
        p_list = []

        for i in range(len(set(self.batchs))):
            q = q_list[i]
            p = q.pow(2) / torch.sum(q, dim=0)
            p = p / torch.sum(p, dim=1, keepdim=True)
            p_list.append(p.detach())

        return p_list


    def forward(self, features, edge_index):

        latent = torch.mm(F.elu(self.encoder_0(features, edge_index)), self.encoder_1)

        # transform parameters
        self.decoder_0.linear.data = self.encoder_1.transpose(0, 1).detach()
        self.decoder_1.data = self.encoder_0.linear.transpose(0, 1).detach()

        gene_recon = torch.mm(
            F.elu(self.decoder_0(latent, edge_index, attention=self.encoder_0.attentions)),
            self.decoder_1
        )

        # calculate DEC q
        q_list = []
        for i in range(len(set(self.batchs))):
            q = 1.0 / ((1.0 + torch.sum((latent[str(i) == self.batchs].unsqueeze(dim=1) \
                                         - self.centroids[i]).pow(2), dim=2)) + 1e-6)
            q = q / torch.sum(q, dim=1, keepdim=True)
            q_list.append(q)

        return latent, gene_recon, q_list


    # calculate KL-divergence -> KL(p || q)
    def kl_div_loss(self, p_list, q_list):

        def kl_div(p, q):
            return torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=1))
        
        dec_loss = 0
        for i in range(len(set(self.batchs))):
            dec_loss = dec_loss + kl_div(p_list[i], q_list[i])

        return dec_loss
