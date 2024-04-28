import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import Data

import st_datasets as stds
from stMSA.integration import stIntegration
from stMSA.aggr_conv import get_micro_emb
from stMSA.utils import set_seed, gen_clust_embed, get_mnn_pairs


def train_integration(
        adata, 
        graph=None,
        radius=None,
        knears=None,
        preclust_method='louvain',
        preclust_para=1,
        dims=[512, 30], 
        epochs=500, 
        lr=1e-3, 
        seed=0, 
        k=100,
        save_model=None,
        device='cuda'
):

    # set seed
    set_seed(seed)

    # graph construction
    if (not isinstance(graph, np.ndarray)):
        graph = stds.pp.concat_adjacency_matrix(adata_list=[
            adata[str(i) == adata.obs['batch'], :] for i in range(len(set(adata.obs['batch'])))
        ], edge_list=[
            stds.pp.build_graph(adata[str(i) == adata.obs['batch'], :], radius=radius, knears=knears) 
            for i in range(len(set(adata.obs['batch'])))
        ])

    # get DEC init centroids
    pca_model = PCA(n_components=30, random_state=seed)
    decomposed_x = pca_model.fit_transform(adata.X.todense().A)
    centroids = gen_clust_embed(decomposed_x, adata.obs['batch'], preclust_method, preclust_para, seed, device) 

    # prepare model
    dims.insert(0, adata.X.shape[1])
    model = stIntegration(dims=dims, centroids=centroids, batchs=adata.obs['batch']).to(device)
    get_micro_env = get_micro_emb().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    data = Data(x=torch.FloatTensor(adata.X.todense().A), edge_index=torch.LongTensor([graph[0], graph[1]])).to(device)

    # calculate node-id map and batch node count
    node_id_map = []
    batch_count = [0]
    for i in range(len(set(adata.obs['batch']))):
        node_id_map.append({
            node: id for node, id in enumerate(np.arange(adata.X.shape[0])[str(i) == adata.obs['batch']])
        })
        batch_count.append(batch_count[i] + len(node_id_map[i]))

    model.train()
    for epoch in tqdm(range(epochs)):
        optim.zero_grad()

        latent, gene_recon, q_list = model(data.x, data.edge_index)

        if (0 == epoch % 100):
            p_list = model.target_distribution(q_list)  # update DEC p parameters

            if (0 != epoch):
                # update mnn node pairs
                node_pairs = get_mnn_pairs(latent.cpu().detach().numpy(), node_id_map, k)
            else:
                # init mnn node pairs by PCA
                node_pairs = get_mnn_pairs(decomposed_x, node_id_map, k)

        # gene reconstruction loss
        gene_recon_loss = F.mse_loss(data.x, gene_recon)

        # adjacency matrix reconstruction loss
        cross_graph_loss = F.triplet_margin_loss(
            latent[node_pairs[0]], latent[node_pairs[1]],
            latent[np.random.randint(adata.X.shape[0], size=node_pairs[0].shape)]
        )
        neg_pairs = np.hstack([
            np.random.randint(low=batch_count[i-1], high=batch_count[i], size=batch_count[i]-batch_count[i-1]) 
            for i in range(1, len(batch_count))
        ])
        single_graph_loss = F.triplet_margin_loss(latent, get_micro_env(latent, data.edge_index), latent[neg_pairs])

        dec_kl_loss = model.kl_div_loss(p_list, q_list)   # DEC clustering loss

        loss = gene_recon_loss + cross_graph_loss + single_graph_loss + 0.01 * dec_kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optim.step()

    # save embedding
    model.eval()
    latent, _, _ = model(data.x, data.edge_index)
    adata.obsm['embedding'] = latent.cpu().detach().numpy()

    # save model
    if (save_model):
        torch.save(model, save_model)

    return adata


if ('__main__' == __name__):

    import scanpy as sc

    adata_list = [stds.get_data(stds.get_dlpfc_data, id=i)[0] for i in range(4)]
    adatas = sc.concat(adata_list, label='batch')
    adatas = adatas[:, adata_list[-1].var['highly_variable']]

    adatas = train_integration(adata=adatas, radius=150)

    _ = stds.cl.evaluate_embedding(adatas, len(set(adatas.obs['cluster']))-1)
