## Setting
Window11환경<br/>

`miniconda`설치
```sh
conda create -n pnu_lab python=3.8
conda activate pnu_lab
# 관련 라이브러리 설치
```

Yolov8n-pose.pt 파일 다운로드 프로젝트 루트 폴더에 위치시켜야합니다.


---
## Data

### DVC
현재 원격 스토리지에 저장되어 있지 않음
```sh
dvc add data
git add data.dvc
git commit -m "commit version 1 data"
git tag -a "v0.1" -m "first version dataset"
```

---
추후 실험 환경 도커 이미지로 변환할 것
