# Requirements

## Structure
```sh
data/
└─ train/
    └─ name.jpg
        ...
    └─ scoliosis_v1.csv
```

### scoliosis_v1 columns
열 이름은 ()안의 이름으로 변경해주시면 좋을 것 같습니다.
값이 없는 경우 그냥 셀을 비워두면 됩니다.

1. 일련번호(img_name)
    data/v1/train 의 이미지 이름과 일치해야함
2. 생년월일(birth)
    YYYYMMDD 형식으로 작성
3. 성별(sex)
    남성 : 0, 여성 : 1
4. 사진 촬영일(shoot date)
    YYYYMMDD
5. 나이(age)
6. 타입(Type)
    normal : 0, T : 1, DT : 2, T+L : 3. TC : 4. L(include TL) : 5




