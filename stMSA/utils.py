import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import faiss
import random
import matplotlib.pyplot as plt


# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def gen_clust_embed(data, batchs, method, para, seed, device):
    start_time = time.time()
    cluster_embed = []
    
    for i in range(len(set(batchs))):
        x = data[str(i) == batchs]

        if ('kmeans' == method):
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=para, init='k-means++', n_init=20, random_state=seed)
            pred = kmeans.fit_predict(x)
        elif ('louvain' == method):
            import scanpy as sc
            adata = sc.AnnData(x)
            sc.pp.neighbors(adata, random_state=seed, use_rep='X', n_neighbors=15)
            sc.tl.louvain(adata, resolution=para, random_state=seed)
            pred = adata.obs['louvain'].astype(int).to_numpy()

        df = pd.DataFrame(x, index=range(x.shape[0]))
        df.insert(loc=1, column='labels', value=pred)

        cluster_embed.append(nn.Parameter(torch.FloatTensor(np.asarray(df.groupby("labels").mean())).to(device)))

    print(f'>>> INFO: Finish generate precluster embedding({time.time() - start_time:.3f}s)!')
    return cluster_embed


def find_similar_index(source: np.ndarray, target: np.ndarray, top_k: int=1):
    index = faiss.IndexFlatL2(source.shape[1])

    faiss.normalize_L2(np.float32(source))
    faiss.normalize_L2(np.float32(target)) 
    index.add(np.float32(target))

    return index.search(np.float32(source), top_k)


def get_mnn_pairs(data, node_id_map, top_k):
    start_time = time.time()
    edge_list = [[], []]

    def get_node_pairs(node_i, node_j):
        return {
            (node_id_map[i][node[0]], node_id_map[j][node[1]]) 
            for node in np.vstack((node_i, node_j)).T
        }
        
    for i in range(len(node_id_map)):
        for j in range(i+1, len(node_id_map)):

            # get used graph gene expression data
            data_i = data[list(node_id_map[i].values())]
            data_j = data[list(node_id_map[j].values())]

            # find approx. similar node by faiss
            distance_i2j, indices_i2j = find_similar_index(data_i, data_j, top_k)
            distance_j2i, indices_j2i = find_similar_index(data_j, data_i, top_k)

            # convert node id to actual id
            used_i2j_list = distance_i2j.reshape(-1) <= distance_j2i.reshape(-1).max() * 0.75
            i2j_pairs = get_node_pairs(
                np.array([np.arange(len(node_id_map[i]))]*top_k).T.reshape(-1)[used_i2j_list], 
                indices_i2j.reshape(-1)[used_i2j_list]
            )
            used_j2i_list = distance_j2i.reshape(-1) <= distance_j2i.reshape(-1).max() * 0.75
            j2i_pairs = get_node_pairs(
                indices_j2i.reshape(-1)[used_j2i_list],
                np.array([np.arange(len(node_id_map[j]))]*top_k).T.reshape(-1)[used_j2i_list]
            )

            # find mnn paris and concat with other edges
            mnn_pairs = np.array(list(i2j_pairs & j2i_pairs)).T
            edge_list = np.array([
                np.concatenate((edge_list[0], mnn_pairs[0])),
                np.concatenate((edge_list[1], mnn_pairs[1]))
            ])
    
    print(f'>>> INFO: Finish finding mmn pairs, find {edge_list.shape[1]} mnn node pairs({time.time() - start_time:.3f}s)!')
    return edge_list


# transform spot coordination
def coor_transform(coor, M):
    return np.dot(M, np.hstack((coor, np.array([1] * coor.shape[0]).reshape(-1, 1))).T)


# set the palette for each label
def get_palette(label_list, opacity=1.0, use_cmap_func=None):
    palette = {}
    max_label = np.array(list(label_list)).shape[0]
    map_label_id = {id: label_name for id, label_name in enumerate(list(label_list))}
    
    # set cmp function
    import matplotlib
    if (not isinstance(use_cmap_func, matplotlib.colors.Colormap)):
        if (max_label < 10):
            use_cmap_func = plt.cm.tab10
        elif (max_label < 20):
            use_cmap_func = plt.cm.tab20
        else:
            use_cmap_func = plt.cm.gist_ncar
    assert(max_label <= use_cmap_func.N), '>>> ERROR: The cmp function has fewer colors than the label count'

    for label_id in range(max_label):
        color = use_cmap_func(int(label_id) / (max_label + 1))
        palette[map_label_id[label_id]] = (color[0], color[1], color[2], opacity)

    return palette


def plotting(coor_list, label_list, save_path=None, palette=None, norm_coor=False, spot_size=1, dims='2d', line_list=None, title=None):

    # if input 2d coor, convert to 3d
    if (2 == coor_list[0].shape[0]):
        new_coor_list = [
            np.vstack([coor, np.ones((1, coor.shape[1]))])
            for coor in coor_list
        ]
        coor_list = new_coor_list

    fig = plt.figure()
    if ('3d' == dims):
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    if (None == palette):
        palette = get_palette(np.unique(np.hstack(label_list)))

    if (norm_coor):
        for i, coor in enumerate(coor_list):
            xs, ys, _ = np.array(coor)
            coor_list[i][0] = xs - xs.min()
            coor_list[i][1] = ys - ys.min()

    for i, coor in enumerate(coor_list):
        xs, ys, _ = np.array(coor)

        label_color = [palette[label] for label in label_list[i]]
        if ('3d' == dims):
            ax.scatter(xs=xs, ys=ys, s=spot_size, zs=i, c=label_color, label=f'slice_{i}')
        else:
            ax.scatter(x=xs, y=ys, s=spot_size, c=label_color, label=f'slice_{i}')
    
    if (isinstance(line_list, np.ndarray)):
        layer = 0
        for lines in line_list:
            xs_0, ys_0, _ = np.array(coor_list[layer])
            xs_1, ys_1, _ = np.array(coor_list[layer+1])

            for line in lines:
                src, dst = line
                ax.plot([xs_0[src], xs_1[dst]], [ys_0[src], ys_1[dst]], [layer, layer+1], c='gray', linestyle='-', linewidth=0.1)
            
            layer += 1

    ax.set_xticks([])
    ax.set_yticks([])
    if ('3d' == dims):
        ax.set_zticks([])
    if (title):
        plt.title(title)

    if (save_path):
        plt.savefig(save_path, dpi=600)
