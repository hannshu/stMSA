from torch_geometric.nn import MessagePassing


class get_micro_emb(MessagePassing):

    def __init__(self):

        super().__init__(aggr='mean')


    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


    def message(self, x_j):
        return x_j
    