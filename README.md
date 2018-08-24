## FocalLossLayer

### Introduction

FocalLossLayer is midified from SoftmaxWithLossLayer.
More info please refer to paper: Focal Loss for Dense Object Detection. 
If it helps your research, Please consider give me a Star.

### Add to caffe

1. Clone this repository
    ```Shell
    git clone https://github.com/liaomingg/Focal-Loss.git
    ```

2. Modify your caffe.proto
    ```
    // Add the next contents to message LayerParameter
    message LayerParameter {
        optional FocalLossParameter focal_loss_param = 151; // select a id.
    }

    // Add the next contents to your caffe.proto
    // Message that stores parameter used by FocalLossLayer
    message FocalLossParameter {
        // loss = -alpha * (1 - pk)^gamma * ln(pk)
        // alpha is a parameter which scale the loss
        optional float alpha = 1 [default = 0.25];
        optional float gamma = 2 [default = 2.00];
    }
    ```

3. Add source file to your Caffe
    Add focal_loss_layer.hpp to Caffe/include/caffe/layers/
    Add focal_loss_layer.cpp to Caffe/src/caffe/layers/
    Add focal_loss_layer.cu to Caffe/src/caffe/layers/

4. Compile your Caffe

### How to use it ?

1. Write codes such as follows in your prototxt file.
    ```
    layer {
        name: "focal_loss"
        type: "FocalLoss"
        bottom: "conv_cls"
        bottom: "label"
        top: "loss"
        include {
            phase: TRAIN
        }
        loss_param {
            ignore_label: 255
        }
        focal_loss_param {
            alpha: 0.25
            gamma: 2.00
        }
    }
    ```
