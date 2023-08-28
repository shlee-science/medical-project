# 환경
CPU : Intel Xeon 2.8GHZ, 32core
OS : CentOS Linux 7
Memory : 362GB
Swp : 64GB
GPU : Nvida A100-PCIE-40GB

_왠지 모르겠지만 nvcc 버전을 다르게 설치해도 torch.cuda가 gpu인식하고 잘 동작함_

## 환경설정

> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
> conda install -c conda-forge albumentations
> conda install -c anaconda scikit-learn
> conda install -c anaconda seaborn
> conda install -c conda-forge tqdm
> conda install -c conda-forge timm
> conda install -c conda-forge ultralytics
> conda install -c anaconda libgcc
> conda install -c conda-forge gcc
> conda install -c conda-forge libstdcxx-ng

> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/lab/lib
> conda install libnvjpeg-dev -c nvidia
> 이외의 필요한 라이브러리 설치

# TODO
- [ ] 데이터 셋 배경 제거 및 합성
- [ ] 타입별 예측(multi class)
- [ ] vision + tabular (multi modal)
- [ ] 레이블 없는 부분 확인 필요 및 tabular 데이터 사용가능한지 확인
- [ ] 이미지 3 ~ 5 가량 회전되어 있는 이미지가 있어서 augmentation 적용 필요

---

**[관련 논문](https://www.nature.com/articles/s42003-019-0635)**
> 논문에서는 Object Detection 관련 Task로 학습할 수 있게 되어있음
> ### 간단한 과정
> 부산대학병원에서 받은 labeling 데이터는 type만 나타내져 있으므로
> 
> 1. 학습된 Yolov8 프레임워크의 딥러닝 모델을 활용해 pose keypoint를 찾고 bbox를 탐색(_augmentation/sample_pipeline.ipynb_)
> 2. efficientnet과 같은 cnn based model을 선정후 학습
> 3. 저장된 모델을 통해 추론
>    - [테스트 영상 1](https://drive.google.com/file/d/1WDoIKtjJyNNEP80vvJB8RZYELtwyms_q/view?usp=sharing)
>    - [테스트 영상 2](https://drive.google.com/file/d/1-uthREs-MrpOqvWeWYueTpI6j1rHnPxW/view?usp=sharing)
