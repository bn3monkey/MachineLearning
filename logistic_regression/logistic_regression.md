# 3.4 로지스틱 회귀
## 3.4.1 계단 함수와 시그모이드 함수

### 시그모이드 함수
<p align="center"><img src="/logistic_regression/tex/f0f64fa9393c8286f37f76c00f2befca.svg?invert_in_darkmode&sanitize=true" align=middle width=100.29432105pt height=34.3600389pt/></p>

- 출력값을 0, 1이 아닌 확률로서 나타내기 위해 도입

![3.4.1](image/1.png)

- 미분값을 자기 자신을 이용하여 표현할 수 있음.
  
  <p align="center"><img src="/logistic_regression/tex/e4b0596e32cf11b32ca658ce0e45a8ee.svg?invert_in_darkmode&sanitize=true" align=middle width=164.1152502pt height=17.2895712pt/></p>

## 3.4.2 모델화

### 3.4.2.1 우도함수와 교차 엔트로피 오차 함수

- 뉴런의 발화 여부 모델링
  - 뉴런이 발화할 확률 `(C=1)`  
  예 : 개인가?

    $$ p(C=1|x) = \sigma(w^Tx+b) $$

  - 뉴런이 발화하지 않을 확률 `(C=0)`  
  예 : 개가 아닌가?

     $$ p(C=0|x) = 1 - p(C=1|x) $$

  - 뉴런이 발화할 확률 `(C=t)`
  
      $$ y = \sigma(w^T+b) $$

      라고 할 때  
      (C=1)이면 y이고, (C=0)이면 1-y이므로,  
      아래와 같이 표현할 수 있다.

      $$p(C=t|x) = y^t(1-y)^{1-t} $$

  - 이 때 입력 데이터가 N개 존재하여, 그 값을 각각 <img src="/logistic_regression/tex/001fd25c5a52c6b4d381f8c071581bd5.svg?invert_in_darkmode&sanitize=true" align=middle width=108.04967414999999pt height=24.65753399999998pt/>이라고 하자.
  - 이 때 출력 데이터도 N개 존재하고, 그 값을 <img src="/logistic_regression/tex/27413cd33c6f718117d8fb364284f787.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06212004999999pt height=20.221802699999984pt/>이라고 하자.
  - N개의 입력 데이터에 대한 총 뉴런의 발화 확률은 아래와 같다.
  - 
  <p align="center"><img src="/logistic_regression/tex/fb87de8819758e1449cba05b722d31ee.svg?invert_in_darkmode&sanitize=true" align=middle width=511.09985520000004pt height=47.60747145pt/></p>
  
  - <img src="/logistic_regression/tex/9d5b7e3db8e1f4e1fd3f34a3f899b21a.svg?invert_in_darkmode&sanitize=true" align=middle width=178.0820877pt height=124.74886710000001pt/>$ E(w,b) = -logL(w,b) = - \sum_{n=1}^{N} \{t_nlogy_n + (1-t_n)log(1-y_n)\} $$

    위와 같은 형태의 함수를 **교차 엔트로피 오차 함수** 라고 한다.  
    줄여서 **오차 함수(error function)** 또는 **손실 함수(loss function)**라고 한다.
    