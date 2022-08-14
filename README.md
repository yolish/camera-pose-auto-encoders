## Camera Pose Auto-Encoders (PAEs)
Official PyTorch implementation of pose auto-encoders for camera pose regression.

This reposistory implements:
1. Our ECCV 2022 paper: ["Camera Pose Auto-Encoders for Improving Pose Regression"](https://arxiv.org/abs/2207.05530)
2. Work in progress: relative pose regression with PAEs

### Prerequisites
In order to run this repository you will need:
1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
1. You can also download pre-trained models to reproduce reported results (see below)
Note: All experiments reported in our paper were performed with an 8GB 1080 NVIDIA GeForce GTX GPU
1. For a quick set up you can run: pip install -r requirments.txt 

### Pre-trained models
| Model (Linked) | Description | 
--- | ---
| APR models ||
[PoseNet+MobileNet](https://drive.google.com/file/d/1YMbWCCnu1pQI-ni4v0y_Lo8EpFumwqHI/view?usp=sharing) | Single-scene APR, KingsCollege scene|
[PoseNet+ResNet50](https://drive.google.com/file/d/1afA97MG5tGfFLBTwaAbauT2-H0tFJxaj/view?usp=sharing}) | Single-scene APR, KingsCollege scene|
[PoseNet+EfficientB0](https://drive.google.com/file/d/1FdaAFfDtN-XxWl3Ek281Ca_wlerqnBOF/view?usp=sharing) | Single-scene APR, KingsCollege scene|
[MS-Transformer](https://drive.google.com/file/d/1ZEIKQSbZmkSnJwETjACvMbs5OeCn7f3q/view?usp=sharing) | Multi-scene APR, CambridgeLandmarks dataset|
[MS-Transformer](https://drive.google.com/file/d/1Ryn5oQ0zRV_3KVORzMAk99cP0fY2ff85/view?usp=sharing) | Multi-scene APR, 7Scenes dataset|
| Camera Pose Auto-Encoders||
[Auto-Encoder for PoseNet+MobileNet](https://drive.google.com/file/d/1EhGmMMCOlFK3lirYlintw0DEJtRGr3eR/view?usp=sharing) | Auto-Encoder for a single-scene APR, KingsCollege scene|
[Auto-Encoder for PoseNet+ResNet50](https://drive.google.com/file/d/1nZolEHF_TKDHpBEw7YQhxH4w61HL0oGD/view?usp=sharing) | Auto-Encoder for a single-scene APR, KingsCollege scene|
[Auto-Encoder for PoseNet+EfficientB0](https://drive.google.com/file/d/1zvHBt5bGB2v59_ONWwLmh8NNUNVdh03T/view?usp=sharing) | Auto-Encoder for a single-scene APR, KingsCollege scene|
[Auto-Encoder for Auto-Encoder for MS-Transformer](https://drive.google.com/file/d/1rshdruRQcZYMIRI9lTY_U981cJsohauI/view?usp=sharing) | Auto-Encoder for a multi-scene APR, CambridgeLandmarks dataset|
[Auto-Encoder for MS-Transformer](https://drive.google.com/file/d/1hGcII8D0G24DBGXh3aLohCubAmfN9Rc7/view?usp=sharing) | Auto-Encoder for a multi-scene APR, 7Scenes dataset|
| Decoders for Image Reconstruction | |
[Decoder for MS-Transformer PAE](https://drive.google.com/file/d/1okm_sN_JXrSD2bpTHBYDghl99pIj87YX/view?usp=sharing) | Decoder trained for reconstructing images from the Shop Facade scene |






 
  
  
  
  
