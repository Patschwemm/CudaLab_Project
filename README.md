# CudaLab Project

Github of Sebastian, Patrick and Sergej for CudaVision Lab - Final Project.

https://www.ais.uni-bonn.de/WS2223/CudaLab/

To Train with Coco dataset download cocoapi from: 
    - https://github.com/cocodataset/cocoapi
Follow install from given github for python.

### TO-DO subtasks

- [x] Implement Accuracy correctly (mAcc)
- [x] Implement mIoU -> Metric apparently correctly implemented, but training on COCO doesn't show much improvement. Bug? Or requires more training?
- [x] City-scapes dataset sequencing
    - [x] Pytorch dataset implementation -> Sort and sequence
    - [x] 5 options for the ground truth (Since it's 5 frame sequences)
- [x] Temporal Augmentation
- [x] Get baseline results (Frame by frame)
- [x] Get results sequenced manner
- [x] ConvGRU, ConvLSTM instead of ConvRNN for the COCO
- [x] Encoder-Decoder (Resnet & VGG-Like)
- [x] Check "temporal regularization"
- [x] (Optional) U-Net (DRU and SRU) 
- [x] Generate images and notebook for results
