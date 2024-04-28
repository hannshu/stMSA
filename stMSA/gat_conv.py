import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels):

        super().__init__(node_dim=0, aggr='add')

        # trainable parameters
        self.linear = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, 1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, 1, out_channels))

        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_normal_(self.linear.data, gain=1.414)
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)


    def forward(self, x, edge_index, attention=None):

        x_src = x_dst = torch.mm(x, self.linear).unsqueeze(dim=1)
        x = (x_src, x_dst)

        if (None == attention):
            # init attention
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            # use attention from encoder layer
            alpha = (attention[0].detach(), attention[1].detach())

        embed = self.propagate(edge_index, x=x, alpha=alpha)
        embed = embed.mean(dim=1)

        return embed


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        alpha = torch.sigmoid(alpha_j + alpha_i)
        alpha = softmax(alpha, index, ptr, size_i)
        embed = x_j * alpha.unsqueeze(dim=-1)

        return embed
