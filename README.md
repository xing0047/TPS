# Domain Adaptive Video Segmentation via Temporal Pseudo Supervision

## Abstract
Video semantic segmentation has achieved great progress under the supervision of large amounts of labelled training data. However, domain adaptive video segmentation, which can mitigate data labelling constraint by adapting from a labelled source domain toward an unlabelled target domain, is largely neglected. We design temporal pseudo supervision (TPS), a simple and effective method that explores the idea of consistency training for learning effective representations from unlabelled target videos. Unlike traditional consistency training that builds consistency in spatial space, we explore consistency training in spatiotemporal space by enforcing model consistency across augmented video frames which helps learn from more diverse target data. Specifically, we design cross-frame pseudo labelling to provide pseudo supervision from previous video frames while learning from the augmented current video frames. The cross-frame pseudo labelling encourages the network to produce high-certainty predictions which facilitates consistency training with cross-frame augmentation effectively. Extensive experiments over multiple public datasets show that TPS is simpler to implement, much more stable to train, and achieves superior video segmentation accuracy as compared with the state-of-the-art.

## Main Results
#### SYNTHIA-Seq => Cityscapes-Seq
| Methods       | road | side. | buil. | pole | light | sign | vege. | sky | per. | rider | car | mIoU |
|---------------|------|-------|-------|------|-------|------|-------|-----|------|-------|-----|------|
| Source        | 56.3 | 26.6 | 75.6 | 25.5 |  5.7 | 15.6 | 71.0 | 58.5 | 41.7 | 17.1 | 27.9 | 38.3 |
| DA-VSN        | 89.4 | 31.0 | 77.4 | 26.1 |  9.1 | 20.4 | 75.4 | 74.6 | 42.9 | 16.1 | 82.4 | 49.5 |
| PixMatch      | 90.2 | 49.9 | 75.1 | 23.1 | 17.4 | 34.2 | 67.1 | 49.9 | 55.8 | 14.0 | 84.3 | 51.0 |
| **TPS**       | **91.2**| **53.7** | 74.9 | 24.6 | **17.9** | **39.3** | 68.1 | 59.7 | **57.2** | **20.3** | **84.5** | **53.8** |

#### VIPER => Cityscapes-Seq
| Methods       | road | side. | buil. | fence | light | sign | vege. | terr. | sky | per. | car | truck | bus | motor | bike | mIoU |
|---------------|------|-------|-------|-------|-------|------|-------|-------|-----|------|-----|-------|-----|-------|------|------|
| Source        | 56.7 | 18.7 | 78.7 |  6.0 | 22.0 | 15.6 | 81.6 | 18.3 | 80.4 | 59.9 | 66.3 |  4.5 | 16.8 | 20.4 | 10.3 | 37.1 |
| PixMatch      | 79.4 | 26.1 | 84.6 | 16.6 | 28.7 | 23.0 | 85.0 | 30.1 | 83.7 | 58.6 | 75.8 | 34.2 | 45.7 | 16.6 | 12.4 | 46.7 |
| DA-VSN        | 86.8 | 36.7 | 83.5 | 22.9 | 30.2 | 27.7 | 83.6 | 26.7 | 80.3 | 60.0 | 79.1 | 20.3 | 47.2 | 21.2 | 11.4 | 47.8 |
| **TPS**       | 82.4 | **36.9** | 79.5 | 9.0 | 26.3 | **29.4** | 78.5 | 28.2 | 81.8 | **61.2** | **80.2** | **39.8** | 40.3 | 28.5 | 31.7 | **48.9** |

**Note: PixMatch is reproduced with replacing the image segmentation backbone to a video segmentaion one.**

## Environment
The code is developed based on PyTorch. All experiments are done on a single 2080Ti GPU. Other platforms or multiple GPUs are not tested.

## Installation
1. create conda environment
```bash
conda create -n TPS python=3.6
conda activate TPS
conda install -c menpo opencv
pip install torch==1.2.0 torchvision==0.4.0
```
2. clone the [ADVENT repo](https://github.com/valeoai/ADVENT)
```bash
git clone https://github.com/valeoai/ADVENT
pip install -e ./ADVENT
```

3. clone the repo
```bash
git clone https://github.com/xing0047/TPS.git
pip install -e ./TPS
```

## Data Preparation
1. [Cityscapes-Seq](https://www.cityscapes-dataset.com/)
```
TPS/data/Cityscapes/
TPS/data/Cityscapes/leftImg8bit_sequence/
TPS/data/Cityscapes/gtFine/
```

2. [VIPER](https://playing-for-bencVhmarks.org/download/)
```
TPS/data/Viper/
TPS/data/Viper/train/img/
TPS/data/Viper/train/cls/
```

3. [Synthia-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar)
```
TPS/data/SynthiaSeq/
TPS/data/SynthiaSeq/SEQS-04-DAWN/
```

## Pretrained Models
Download [here]() and put them under pretrained\_models.

## Optical Flow Estimation
- Synthia-Seq \\
  [train]()
- VIPER \\
  [train](https://drive.google.com/drive/folders/1i_-yw9rS7-aa7Cn5ilIMbkUKwr1JpUFA?usp=sharing)
- Cityscapes-Seq \\
  [train]() | [val]()

## Train and Test

## Evaluation

## Acknowledgement
This codebase is heavily borrowed from [DA-VSN](https://github.com/Dayan-Guan/DA-VSN).

## Contact
If you have any questions, feel free to contact: xing0047@e.ntu.edu.sg
