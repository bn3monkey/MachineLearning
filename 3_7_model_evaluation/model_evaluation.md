# 3.7 모델 평가
## 3.7.1 분류에서 예측으로

![3.7.1](image/1.png)

- 실제 세계에서는 다음과 같이 노이즈가 섞여있거나 겹쳐진 데이터가 존재할 수 있다. 이와 같은 데이터들이 주어졌을 때 어떻게 분리하여야 할까?

![3.7.2](image/2.png)

- 오른쪽의 분류가 가장 이상적인 분류이다.
- 왼쪽의 분류 역시 '잘 분류했다'고 판단하는 것은 데이터의 경향성을 잘 파악하여 분류하였기 때문이다.
- 비슷한 데이터가 주어졌을 때, 왼쪽처럼 **데이터의 경향성을 어느 정도 잘 분류할 수 있는 법**을 생각해야 한다.

## 3.7.2 예측을 평가
![3.7.3](image/3.png)

- 현재 모델이 비슷한 데이터에서도 잘 분류하는 지를 확인하기 위하여,  
  전체 데이터셋에서 훈련용 데이터와 테스트 데이터로 나누고, 훈련용 데이터로 훈련을 진행한 뒤 테스트 데이터로 평가한다.

### 혼합 행렬
모델이 예측한 값을 y, 실제 값을 t라고 하였을 때, y와 t의 결과에 따라 가능한 조합으로 나타낸 것이 아래의 **혼합 행렬**이다.

|| <img src="/3_7_model_evaluation/tex/0ae115a65fe296fc4641cc1190e57d4a.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/> | <img src="/3_7_model_evaluation/tex/a42b1c71ca6ab3bfc0e416ac9b587993.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/>|
|-|-|-|
|<img src="/3_7_model_evaluation/tex/ea8e02b76558beb2e7fbd75146337fe7.svg?invert_in_darkmode&sanitize=true" align=middle width=36.07293689999999pt height=21.18721440000001pt/>|TP (True positive, 진양성)|FN(False Negtive, 위음성)|
|<img src="/3_7_model_evaluation/tex/1c899e1c767eb4eac89facb5d1f2cb0d.svg?invert_in_darkmode&sanitize=true" align=middle width=36.07293689999999pt height=21.18721440000001pt/>|FP (False postivie, 위양성)|TN(True Negative, 진음성)|

### 모델 평가 지표
위의 혼합 행렬에 있는 값을 이용하여, 모델의 의미를 판단하는 지표는 아래와 같다.

|명칭|식|설명|
|-|-|-|
|정답률(Accuracy)|<img src="/3_7_model_evaluation/tex/298ccc87cb0f1d16585fb5772bd49d01.svg?invert_in_darkmode&sanitize=true" align=middle width=113.12369805000002pt height=28.670654099999997pt/>|전체 데이터 중 예측 값과 실제 값이 맞은 비율|
|적합률(Precision)|<img src="/3_7_model_evaluation/tex/f4cd9bcb5a7ec633e178925192d28465.svg?invert_in_darkmode&sanitize=true" align=middle width=50.00875934999999pt height=28.670654099999997pt/>|모델에서 발화한 데이터<img src="/3_7_model_evaluation/tex/c85ec04d1975644fb778ab52df5c2e7e.svg?invert_in_darkmode&sanitize=true" align=middle width=51.571479299999986pt height=24.65753399999998pt/> 중 실제로 발화해야 됐던 데이터<img src="/3_7_model_evaluation/tex/7701a0d84da02e9d96eed41a60082947.svg?invert_in_darkmode&sanitize=true" align=middle width=94.95030104999998pt height=24.65753399999998pt/>의 비율|
|재현률(Recall)|<img src="/3_7_model_evaluation/tex/afdc989aa28ac82e2cd6dcf4fdd711c0.svg?invert_in_darkmode&sanitize=true" align=middle width=51.51617294999999pt height=28.670654099999997pt/>|모델에서 발화해야 하는 데이터<img src="/3_7_model_evaluation/tex/e5550fe4d0135e29f6f5d196796c4ee9.svg?invert_in_darkmode&sanitize=true" align=middle width=48.858371099999985pt height=24.65753399999998pt/> 중 실제로 발화한 데이터<img src="/3_7_model_evaluation/tex/7701a0d84da02e9d96eed41a60082947.svg?invert_in_darkmode&sanitize=true" align=middle width=94.95030104999998pt height=24.65753399999998pt/>의 비율|

### 평가에 대한 예시

![3.7.4](image/4.png)

- 사진을 주어주고 개를 판별하는 모델을 만들었다고 가정해보자.
- 위와 같이 6마리의 개와 4마리의 고양이가 있는 테스트 데이터를 모델에 입력했을 경우를 생각해본다.
- 이 모델은 4마리의 개와 1마리의 고양이를 개라고 판별한다.
- 이 모델은 3마리의 고양이와 2마리의 개를 개가 아니라고 판별했다.
- 위의 모델을 혼합행렬로 나타냈을 때 아래와 같이 볼 수 있다.

|조합|데이터 개수|
|-|:-:|
|진양성|4|
|위양성|1|
|진음성|3|
|위음성|2|

- 위의 모델을 지표로 판단했을 때, 다음과 같은 퍼센티지를 보여준다.

|지표|퍼센트|계산|
|-|:-:|-|
|정답률(Accuracy)|70%|<img src="/3_7_model_evaluation/tex/e2f426a5e144cec92088ec129cc68159.svg?invert_in_darkmode&sanitize=true" align=middle width=56.4843081pt height=27.77565449999998pt/>|
|적합률(Precision)|80%|<img src="/3_7_model_evaluation/tex/dbd38b49106fe0cd9846036457e887f7.svg?invert_in_darkmode&sanitize=true" align=middle width=23.196467249999994pt height=27.77565449999998pt/>|
|재현률(Recall)|66.6%|<img src="/3_7_model_evaluation/tex/8c2445f45a2492167e2a70f61dba34a4.svg?invert_in_darkmode&sanitize=true" align=middle width=23.196467249999994pt height=27.77565449999998pt/>|

## 3.7.3 실험
