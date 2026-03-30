# Energy Benchmark Study

This repository contains the code used in the benchmark study [1], which investigates scaling behavior of ResNet-50 [2] trained on ImageNet-2012 [3].
Parallization parameters - local batch size (LBS), global batch size (GBS), and GPU count - were varied while tracking the energy consumption during training. 
The model and data loader were used from torch, energy measurements were conducted using the python program package *[perun](https://github.com/Helmholtz-AI-Energy/perun)* [4].  
This code is provided for reproducibility and comparison, not as a general-purpose training framework.

## Repository Structure
+ `scripts/`: Entry point (`main.py`) for training and scripts for post-processing and evaluation of raw results.
+ `resnet/`: Core ResNet implementation, including training and evaluation of raw data.
+ `submission_scripts/`: Job submission scripts for running experiments on the HPC system.


## Data
The model was trained on ImageNet-2012 [3] using 1281167 training samples and 50000 validation samples distributed among 1000 classes.

## Environment
See pyproject.toml.
```bash 
pip install -e .
```

## Experiments

### Energy Consumption During Training

Two approaches of scaling experiments were performed: Maintaining a constant dataset size while increasing the GPU count and a linearly increasing dataset size with increasing GPU count.
For both, the GBS or LBS was scaled while scaling the GPU count.
The training time, Top1 error, and energy consumption were tracked.

### Time Profiling of Training Phases

Timestamps are inserted to measure the duration of data loading, forward pass, and backward pass.
Data loading is decomposed into transfer from the data source to CPU memory and from CPU to GPU memory.
When data staging (i.e., prefetching of subsequent batches) is enabled, the reported data-loading time corresponds to the interval from the start of a batch iteration until all data required for that iteration is available on the GPUs.
Consequently, the time spent loading data staged in previous iterations is not included.
Data staging is enabled for all benchmark experiments.


## References

[1] P. Huber, D. Li, J. P. Gutiérrez Hermosillo Muriedas, D. Kieckhefen, M. Götz, A. Streit, C. Debus,
"Energy Consumption in Parallel Neural Network Training", 
*Proceedings of the 2026 SIAM Conference on Parallel Processing for Scientific Computing (PP)*,
2026, 46--59,
[doi: 10.1137/1.9781611979022.4](https://doi.org/10.1137/1.9781611979022.4)  
[2] K. He, X. Zhang, S. Ren and J. Sun, 
"Deep Residual Learning for Image Recognition",
*2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, 770-778, [doi: 10.1109/CVPR.2016.90](http://doi.org/10.1109/CVPR.2016.90).  
[3] J. Deng, W. Dong, R. Socher, L. -J. Li, Kai Li and Li Fei-Fei, 
"ImageNet: A large-scale hierarchical image database", 
*2009 IEEE Conference on Computer Vision and Pattern Recognition*, Miami, FL, USA, 2009, 248-255, [doi: 10.1109/CVPR.2009.5206848](http://doi.org/10.1109/CVPR.2009.5206848).  
[4] J. P. Gutiérrez Hermosillo Muriedas, K. Flügel, C. Debus, H. Obermaier, A. Streit, and M. Götz, 
"perun: Benchmarking energy consumption of high-performance computing applications",
*European Conference on Parallel Processing*, Springer, 2023, 17–31, [doi: 10.1007/978-3-031-39698-4_2](https://doi.org/10.1007/978-3-031-39698-4_2).

