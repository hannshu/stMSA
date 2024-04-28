import numpy as np
import torch
from torch_geometric.data import Data
import st_datasets as stds


def eval_integration(
        adata,
        model_path,
        graph=None,
        radius=None,
        knears=None,
        device='cuda'
):

    if (not isinstance(graph, np.ndarray)):
        graph = stds.pp.concat_adjacency_matrix(adata_list=[
            adata[str(i) == adata.obs['batch'], :] for i in range(len(set(adata.obs['batch'])))
        ], edge_list=[
            stds.pp.build_graph(adata[str(i) == adata.obs['batch'], :], radius=radius, knears=knears) 
            for i in range(len(set(adata.obs['batch'])))
        ])

    model = torch.load(model_path).to(device)
    data = Data(x=torch.FloatTensor(adata.X.todense().A), edge_index=torch.LongTensor([graph[0], graph[1]])).to(device)

    print(f'>>> INFO: Successfully load model.')
    model.eval()
    latent, _, _ = model(data.x, data.edge_index)
    adata.obsm['embedding'] = latent.cpu().detach().numpy()

    return adata


if ('__main__' == __name__):
    
    import scanpy as sc

    adata_list = [stds.get_data(stds.get_dlpfc_data, id=i)[0] for i in range(4)]
    adatas = sc.concat(adata_list, label='batch')
    adatas = adatas[:, adata_list[-1].var['highly_variable']]

    adatas = eval_integration(adatas, '/root/stMSA_paras/DLPFC/151507-151510.pt', radius=150)

    _, score = stds.cl.evaluate_embedding(adatas, n_cluster=7)
