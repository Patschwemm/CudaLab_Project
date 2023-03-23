# CudaLab Project

Github of Sebastian, Patrick and Sergej for CudaVision Lab - Final Project.

https://www.ais.uni-bonn.de/WS2223/CudaLab/
Access to CudaLab Folder on Website with:
 - Username: VisionLabWS2223
 - Password: ConvNextB

To access lab cuda gpus:
 - e.g. ssh villar@cuda3.informatik.uni-bonn.de
 - Machines available : cuda3, cuda7, cuda9, cuda10, cuda11, cuda12


To Train with Coco dataset download cocoapi from: 
    - https://github.com/cocodataset/cocoapi
Follow install from given github for python.

### TO-DO subtasks

- [x] Implement Accuracy correctly (mAcc)
- [x] Implement mIoU -> Metric apparently correctly implemented, but training on COCO doesn't show much improvement. Bug? Or requires more training?
- [x] City-scapes dataset sequencing
    - [x] Pytorch dataset implementation -> Sort and sequence
    - [x] 5 options for the ground truth (Since it's 5 frame sequences)
- [] Temporal Augmentation
- [x] Get baseline results (Frame by frame)
- [] Get results sequenced manner
- [] ConvGRU, ConvLSTM instead of ConvRNN for the COCO
- [] Encoder-Decoder (Resnet & VGG-Like)
- [] Check "temporal regularization"
- [] (Optional) U-Net (DRU and SRU) 
- [] Generate images and notebook for results
