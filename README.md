An efficient python package for Interface Simulation.
1. Visualizing lattice matching information by polar projection figure;
2. Symmetry analysis to screen out identical matching and termination conditions;
3. Structure pre-optimization by MLIP-predicted interface energy.

Install
`pip install .`

InterOptimus integrated machine learning interatomic potentials (MLIPs) including `grace-2l` `chgnet` `mace` `orb-models` `sevenn`. Details can be find in https://matbench-discovery.materialsproject.org/

As these MLIP packages are not compatible in the same python environment, to use them flexibly to calculate atomic energies of interface structures in InterOptimus, we created their corresponding docker images with their required python environments and packages respectively, and achieve usages of these MLIPs through requesting their docker containers for energy prediction results. Our images are uploaded in the Alibaba Cloud. Therefore, to use our package, you need to register an Alibaba Cloud account at https://account.alibabacloud.com/ and install docker.

After you register your Alibaba Cloud account, go to the `Container Registry/Instances` page, follow the instruction to make a totally free `Instance of Personal Edition`, and get your countainer registry [username] and [password] which you will need to login in to the docker registry.

![image](https://github.com/user-attachments/assets/bd4240f8-f9d2-4f36-990b-579963a7462a)

Finally, execute the `docker login` command provided in your own `Container Registry/Instances` page.
