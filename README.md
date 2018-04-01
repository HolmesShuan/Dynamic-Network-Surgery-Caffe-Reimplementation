# Dynamic-Network-Surgery-Caffe-Reimplementation
Caffe reimplementation of dynamic network surgery(GPU-only/cuDNN unsupported yet).<br>
Official version at [Here (https://github.com/yiwenguo/Dynamic-Network-Surgery).](https://github.com/yiwenguo/Dynamic-Network-Surgery)<br>
## Main Differences:
* We `didn't` prune the bias term.
* We make the selection of hyperparameter clear and convincing. (May `differ` from official version but more `readable`)
* We re-adjust the organization of the code.<br> 
You may `monitor the change of weights sparsity` of convolutional layers and innerproduct layers.
* We re-write the original convolution layer and innerproduct layer instead of creating new classes.<br>
It will be easier to reuse the exisiting `.prototxt` of standard Caffe framework. 

## How to use ?
The sames as the standard Caffe framework.<br>
```
$ make all -j8 # USE_NCCL=1 make all -j8 for multi-GPU support
$ ./build/tools/caffe train --weights /ModelPath/Ur.caffemodel --solver /SolverPath/solver.prototxt -gpu 0
$ ./build/tools/caffe test --weights /ModelPath/Ur.caffemodel --model /StructPath/train_val.prototxt -gpu 0 -iterations 100
# Please notice: 
# CPU Version is not supported yet, but you may find it quite easy to rewrite conv_layer.cpp and innerproduct_layer.cpp from .cu files.
```
You can load pre-trained caffemodel into this framework to fine-tune (strongly recommend) or re-train from the begining (Remember to set the `threshold` in `train_val.prototxt`, which will be mentioned below).

### Usage Example :
Pre-trained Caffemodel:<br>
[AlexNet with BN (https://github.com/HolmesShuan/AlexNet-BN-Caffemodel-on-ImageNet)](https://github.com/HolmesShuan/AlexNet-BN-Caffemodel-on-ImageNet)<br>
Sparse (50%) convolution layers should `outperform` baseline.

**Pruned Layer**<br>
```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  # weight_mask param
  param {
    ## Indispensable !!!!
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    pad: 2
    threshold: 0.6 ## based on the 68-95-99.7 rule [defalut 0.6]
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    # weight_mask_filler { ## omissible 
    #   type: "constant"
    #   value: 1 ## This term has been reset to 1 in caffe.proto
    # }
  }
}
```
**Dense Layer**
```
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    sparsity_term: false ## Default is true
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
```
**solver.prototxt**
```
net: "models/bvlc_alexnet/train_val.prototxt"
base_lr: 0.001
lr_policy: "multistep"
gamma: 0.1
stepvalue: 84000
display: 20
max_iter: 162000
momentum: 0.9
weight_decay: 0.00005
snapshot: 6000
snapshot_prefix: "models/bvlc_alexnet/alexnet-BN"
solver_mode: GPU
surgery_iter_gamma: 0.0001 ## [default 1e-4] Probability(do surgery) = (1+gamma*iter)^-power 
surgery_iter_power: 1 ## [default 1] 
```
## Tips
1. The selection of threshold is pretty tricky. It may differ a lot between different layers.
2. If you encounter the vanishing gradient problem, then you could adjust `gamma` and `power` in `solver.prototxt`. <br>
Multiple attempts failed, then reduce `threshold`. 

Threshold | Sparsity
------------ | -------------
0.674 | 50%
0.994 | 68%
1.281 | 80%
1.644 | 90%
1.959 | 95%
 
## Citation
Basic idea comes from:
```
@inproceedings{guo2016dynamic,    
  title = {Dynamic Network Surgery for Efficient DNNs},
  author = {Guo, Yiwen and Yao, Anbang and Chen, Yurong},
  booktitle = {Advances in neural information processing systems (NIPS)},
  year = {2016}
} 
```
And base on Caffe framework:
```
@article{jia2014caffe,
  Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
  Journal = {arXiv preprint arXiv:1408.5093},
  Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
  Year = {2014}
}
```

