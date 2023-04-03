# MultiBA_GCN
MultiBA_GCN for 3d pose estimate, 
We propose Multi Branch Attentions  Graph Convolutional Networks (MultiBA_GCN), the novel graph convolutional operation with graph-structured data. 

### Results on Human3.6M

Under Protocol 1 (mean per-joint position error) and Protocol 2 (mean per-joint position error after rigid alignment).

| Method | 2D Detections | # of Epochs | # of Parameters | MPJPE (P1) | P-MPJPE (P2) |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Martinez et al. [1] | Ground truth | 200  | 4.29M | 44.40 mm | 35.25 mm |
| SemGCN | Ground truth[2] | 30 | 0.27M | 42.14 mm | 33.53 mm |
| SemGCN (w/ Non-local)[2] | Ground truth | 30 | 0.43M | 40.78 |31.46 |    

| MultiBA_GCN (no FC) | Ground truth | 50 | 0.40M | 40.5165 | 32.0645 |
| **MultiBA_GCN (FC,sum fusion)** | Ground truth | 50 | 0.47M | **38.4044 mm** | **30.4082 mm** |


| Martinez et al. [1] | SH (fine-tuned) | 200  | 4.29M | 63.48 mm | 48.15 mm |
| SemGCN (w/ Non-local)[2] | SH (fine-tuned) | 100 | 0.43M | 61.24 mm | 47.71 mm |
| **MultiBA_GCN (FC,conv fusion)** | CPN | 50 | 0.93M | ** 55.1999 mm** | **43.9530 mm** |

Results using two different 2D detections (Ground truth and CPN on Human3.6M) are reported.

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch >= 0.4.0

### Dataset setup
You can follow the instructions for setting up the Human3.6M and results of 2D detections in [`DATASETS.md`](DATASETS.md) from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Also you can download the data file from yun.baidu.com:  XXXXX. 
All the npz file need be put in dataset directory

### Evaluating our pretrained models
The pretrained models can be downloaded from yun.baidu.com.cn:XXXX and put it on your path.
To evaluate our method on gt data, run:
python main_gcn.py  --evaluate yourpath/gt_mbagcn.pth.tar
To evaluate our method on cpn data, run:
python main_gcn.py  --evaluate yourpath/ckpt_semgcn_nonlocal_sh.pth.tar --k cpn

### Training from scratch
If you want to reproduce the results of our pretrained models using gt data, run the following commands:
python main.py --epochs 50
For training and evaluating models using 2D detections generated by fine-tuned CPN detections, add `--k cpn` to the commands:
python main.py --epochs 50 -k cpn

If you want verify the ablation study, modify the in import statement as the prompt information on the main.py heads.

### References
[1] Martinez et al. [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf). ICCV 2017.
[2] Zhao et al. [Semantic Graph Convolutional Networks for 3D Human Pose Regression](https://arxiv.org/pdf/1904.03345.pdf). CVPR 2019.
[3] Pavllo et al. [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf). CVPR 2019.

## Acknowledgement

Part of our code is borrowed from the following repositories.

- [Sem_Gcn](https://github.com/garyzhao/SemGCN)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

We thank to the authors for releasing their codes. Please also consider citing their works.


