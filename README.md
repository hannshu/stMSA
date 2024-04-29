# stMSA
[![Documentation Status](https://readthedocs.org/projects/stmsa/badge/?version=latest)](https://stmsa.readthedocs.io/en/latest/?badge=latest)

![stMSA overview](./framework.png) 

## Overview 
Spatial transcriptomics (ST) is a valuable methodology that integrates spatial location data with gene expression information, generating novel insights for biological research. Given the substantial number of ST datasets available, researchers are becoming more inclined to unveil potential biological features across larger datasets, thus obtaining a more comprehensive perspective. However, existing methods predominantly concentrate on cross-batch feature learning, disregarding the intricate spatial patterns within individual slices. Consequently, effectively integrating features across different slices while considering the slice-specific patterns poses a substantial challenge. To overcome this limitation and enhance the integration performance of multi-slice data, we propose a deep graph-based auto-encoder model incorporating contrastive learning techniques, named stMSA. This model is specifically tailored to generate batch-corrected representations while preserving the unique spatial patterns within each slice. It achieves this by simultaneously considering both inner-batch and cross-batch patterns during the integration process. We observe that stMSA surpasses existing state-of-the-art methods in discerning domain structures and cross-batch tissue structures across different slices, even when confronted with diverse experimental protocols and sequencing technologies. Furthermore, the representations learned by stMSA exhibit outstanding performance in matching two slices in the development dataset of a mouse embryo and aligning multi-slice mouse brain coronal sections.

## Software dependencies
scanpy==1.9.3  
squidpy==1.3.0  
pytorch==1.13.0(cuda==11.6)   
torch_geometric==2.3.1(cuda==11.6)  
R==3.5.1  
mclust==5.4.10

## Setup stMSA
### Setup by Docker (*Recommended*):  
1. Download the stMSA image from [DockerHub](https://hub.docker.com/repository/docker/hannshu/stmsa) and setup a container:
``` bash
docker run --gpus all --name your_container_name -idt hannshu/stmsa:latest
```

2. Access the container:
``` bash
docker start your_container_name
docker exec -it your_container_name /bin/bash
```

3. Write a python script to run stMSA

The anaconda environment for stMSA will be automatically activate in the container. The stMSA source code is located at `/root/stMSA`, please run ```git pull``` to update the codes before you use.
All dependencies of stMSA have been properly installed in this container, including the mclust R package, and the conda environment stMSA will automatically activate when you run the container.

- Note: Please make sure `NVIDIA Container Toolkit` is properly installed on your host device. (Or follow this instruction to [setup NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first)

- Details of the container
``` bash
/root
|-- stMSA           # The stMSA source code
|-- stMSA_paras     # The model parameters of stMSA for each experiment
`-- stMSA_results   # The embedding result of each experiment
```

### Setup by anaconda:  
[install anaconda](https://docs.anaconda.com/free/anaconda/install/)

1. Clone this repository from Github:
``` bash
git clone https://github.com/hannshu/stMSA.git
```

2. Download dataset repository:

``` bash
git submodule init
git submodule update
```

3. Import conda environment:  
``` bash
conda env create -f environment.yml
```

4. Write a python script to run stMSA

- Note: If you need to generate clustering result by mclust, you need to install [mclust](https://github.com/hannshu/st_clustering/blob/master/mclust_package/mclust_5.4.10.tar.gz) package to the R environment in your conda environment.
- If the `environment.yml` file not fit your system or device, please try the [Docker container](https://hub.docker.com/repository/docker/hannshu/stmsa) we provided.

## Example
``` python
import scanpy as sc

from train_integrate import train_integration
import st_datasets as stds

# load data
adata_list = [stds.get_data(stds.get_dlpfc_data, id=i)[0] for i in range(4)]
adatas = sc.concat(adata_list, label='batch')
adatas = adatas[:, adata_list[-1].var['highly_variable']]

# train stMSA
adatas = train_integration(adata=adatas, radius=150)

# calculate the clustering result
adata = stds.cl.evaluate_embedding(adatas, len(set(adatas.obs['cluster']))-1)

# other downstream tasks
# ...
```

## Tutorial
Read the [Documentation](https://stmsa.readthedocs.io/en/latest/) for detailed tutorials.

<!-- ## Citation
If you have found our model useful in your work, please consider citing [our article](url):
```

``` -->