Installation
=====

.. _installation:

Important software Dependencies
::::::

- scanpy==1.9.3
- squidpy==1.3.0
- pytorch==1.13.0(cuda==11.6)
- torch_geometric==2.3.1(cuda==11.6)
- R==3.5.1
- mclust==5.4.10


Setup by Docker (Recommended)
~~~~~~~~~~~~

1. Download the stMSA image from `DockerHub <https://hub.docker.com/repository/docker/hannshu/stmsa>`_ and setup a container:

   .. code-block:: bash

      docker run --gpus all --name your_container_name -idt hannshu/stmsa:latest

2. Access the container:

   .. code-block:: bash

      docker start your_container_name
      docker exec -it your_container_name /bin/bash

3. Write a Python script to run stMSA. 

The anaconda environment for stMSA will be automatically activate in the container. The stMSA source code is located at ``/root/stMSA``, please run ``git pull`` to update the codes before you use.
All dependencies of stMSA have been properly installed in this container, including the mclust R package, and the conda environment stMSA will automatically activate when you run the container.

- Note: Please make sure `NVIDIA Container Toolkit` is properly installed on your host device. (Or follow this instruction to `setup NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ first)

- Details of the container

.. code-block:: bash
   
   /root
   |-- stMSA           # The stMSA source code
   |-- stMSA_paras     # The model parameters of stMSA for each experiment
   `-- stMSA_results   # The embedding result of each experiment

Setup by Anaconda
~~~~~~~~~~~~

1. Install `Anaconda <https://docs.anaconda.com/free/anaconda/install>`_.

2. Clone the stMSA repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/hannshu/stMSA.git

3. Download the dataset repository:

   .. code-block:: bash

      git submodule init
      git submodule update

4. Import the conda environment:

   .. code-block:: bash

      conda env create -f environment.yml

5. Write a Python script to run stMSA.

- Note: If you need to generate clustering result by mclust, you need to install `mclust <https://github.com/hannshu/st_clustering/blob/master/mclust_package/mclust_5.4.10.tar.gz>`_ package to the R environment in your conda environment.
- If the `environment.yml` file not fit your system or device, please try the `Docker container <https://hub.docker.com/repository/docker/hannshu/stmsa>`_ we provided.
