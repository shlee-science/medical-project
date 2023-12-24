## 서버 환경
CPU : Intel Xeon 2.8GHZ, 32core</br>
OS : CentOS Linux 7</br>
Memory : 362GB</br>
Swp : 64GB</br>
GPU : Nvida A100-PCIE-40GB

_왠지 모르겠지만 nvcc 버전을 다르게 설치해도 torch.cuda가 gpu인식하고 잘 동작함_

## 환경설정
```
# 용량 부족 문제로 인해 가상환경 설정을 위해 conda 가상환경을 ~/data/로 옮겨야함(패키지 경로, 가상환경 경로 둘다 바꿔주어야 함)
# 폴더를 생성해주고 아래 명령어들 실행
conda config --append envs_dirs ~/data/conda/envs
conda config --add pkgs_dirs ~/data/conda/pkgs
conda create -n lab python=3.8

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge albumentations
conda install -c anaconda scikit-learn
conda install -c anaconda seaborn
conda install -c conda-forge tqdm
conda install -c conda-forge timm
conda install -c conda-forge ultralytics
conda install -c anaconda libgcc
conda install -c conda-forge libstdcxx-ng
```

> conda install -c conda-forge gcc
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/lab/lib</br>
> conda install libnvjpeg-dev -c nvidia</br>
> 이외의 필요한 라이브러리 설치