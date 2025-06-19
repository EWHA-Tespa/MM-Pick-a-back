# MM-Pick-a-back

### 📁 폴더 구조
```
.
├── data/ 
├── checkpoints_perceiver_io/ # 모델 체크포인트 저장되는 위치
├── logs_perceiver_io/ # baseline.sh 실행 후 log 결과 파일이 저장되는 위치
├── models/ # CPG 학습에 필요한 모델 버전이 저장되는 위치
├── packnet_models/ # Packnet 학습에 필요한 모델 버전이 저장되는 위치
├── packnet_models_pickaback/ # modeldiff 하기 위한 모델이 저장되는 위치
├── run_pickaback/ # 실험 실행 위한 스크립트 파일들이 저장되는 위치
├── CPG_cifar100_main_normal.py # 선행 연구의 CPG 위한 파이썬 파일
├── CPG_MM_main_normal.py # MM-Pick-a-back의 CPG 위한 멀티모달 버전 파이썬 파일
├── transfer_kv_perceiver_io.py # k/v projection 교환 위한 파이썬 파일
├── packnet_cifar100_main_noraml.py # Packnet 학습 위한 파이썬 파일
├── pickaback_perceiverIO.py # perceiver IO 기반 modeldiff 위한 파이썬 파일
├── tools/
├── utils/
└── utils_pickaback/
```

---

## 모듈 실행 방법
### 🟠 레포지토리 clone
```
https://github.com/EWHA-Tespa/MM-Pick-a-back.git
cd MM-Pick-a-back
```
### 🟠 데이터셋 다운로드 및 전처리
__📊 CUB-200-2011:__
```
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 -O CUB_200_2011.tgz # 이미지 다운로드
tar -xvzf CUB_200_2011.tgz # 압축 해제
```
텍스트 데이터 (별도 다운로드): https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view?resourcekey=0-8y2UVmBHAlG26HafWYNoFQ 

전처리 방법: https://github.com/EWHA-Tespa/MM-Pickaback-Data-Preprocess/blob/main/cub.ipynb 의 raw 파일을 데이터셋 저장 위치에 붙여넣고, 해당 파일을 실행하여 전처리합니다. 

__📊 MSCOCO:__
```
wget http://images.cocodataset.org/zips/train2017.zip # 학습 이미지 다운로드 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip # annotation 다운로드
unzip train2017.zip # 압축 해제
unzip annotations_trainval2017.zip # 압축 해제
```

전처리 방법: https://github.com/EWHA-Tespa/MM-Pickaback-Data-Preprocess/blob/main/mscoco/preprocess_0430.ipynb 의 raw 파일을 데이터셋 저장 위치에 붙여넣고, 해당 파일을 실행하여 전처리합니다. 

__📊 Oxford-102-flowers:__

kaggle competitions download -c oxford-102-flower-pytorch

텍스트 데이터 (별도 다운로드): https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?resourcekey=0-Av8zFbeDDvNcF1sSjDR32w

전처리 방법: https://github.com/EWHA-Tespa/MM-Pickaback-Data-Preprocess/blob/main/oxford.ipynb 의 raw 파일을 데이터셋 저장 위치에 붙여넣고, 해당 파일을 실행하여 전처리합니다. 

### 🟠 실행

MM-Pick-a-back 실험 결과는 `run_pickaback/`으로 실행합니다.

__1. Baseline__
```
bash run_pickaback/baseline.sh cub # 사용 데이터셋에 따라 cub, mscoco, oxford로 구분합니다.
```
터미널에 위의 명령어를 실행하면, 'logs_perceiver_io/baseline_cub_acc_scratch.txt' 파일이 생성되고, 'checkpoints_perceiver_io/baseline_scratch'에 모델 체크포인트가 저장됩니다. 

__2-1. Backbone task 학습__

다음 명령어는 첫 번째 작업을 학습하기 위한 것입니다. 첫 번째 작업은 이후 작업을 위한 백본(backbone) 모델로 사용됩니다. 이 명령어는 학습과 가지치기(pruning) 단계를 함께 수행합니다.
```
bash run_pickaback/wo_backbone.sh cub
```
터미널에 위의 명령어를 실행하면, 'checkpoints_perceiver_io/CPG_single_scratch_woexp'에 모델 체크포인트가 저장됩니다. 

__2-2. Pruning ratio__

백본 모델에 대해 적절한 가지치기 비율(pruning ratio)을 선택합니다. 
```
bash run_pickaback/select_pruning_ratio_of_backbone.sh
```

__3. 유사한 모델 탐색__

ModelDiff를 기반으로 대상 작업에 대해 의사결정 패턴이 유사한 모델을 선택합니다.
```
bash run_pickaback/find_backbone.sh
```
대상 작업에 대해 유사한 모델을 다음과 같이 찾습니다:

Selected backbone for target 14 = (euc)4

__4. 백본 재구성__

target모델의 key value projection을 backbone 모델의 것으로 교환하여 이후 크로스-모달 간 학습을 준비합니다.
```
bash run_pickaback/transfer_kv.sh
```

__5. target 모델 학습__

다음 명령어를 실행하여 대상 작업에 대한 지속 학습(continual learning) 모델을 학습합니다.
```
bash run_pickaback/w_backbone_MM.sh
```

터미널에 위의 명령어를 실행하면, 'checkpoints_perceiver_io/CPG_fromsingle_scratch_woexp_target'에 모델 체크포인트가 저장됩니다. 

---

### ✍️ 모델 및 코드 기여

이 프로젝트는 다음 오픈소스에 기반합니다:

* [Pick-a-back: Selective Device-to-Device Knowledge Transfer in Federated Continual Learning](https://github.com/jinyi-yoon/Pick-a-back)
* [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://github.com/lucidrains/perceiver-pytorch)

