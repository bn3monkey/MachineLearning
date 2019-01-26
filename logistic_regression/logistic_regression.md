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

    $$ p(C=1|x) = \sigma(w^{T}x+b) $$

  - 뉴런이 발화하지 않을 확률 `(C=0)`  
  예 : 개가 아닌가?

     $$ p(C=0|x) = 1 - p(C=1|x) $$

  - 뉴런이 발화할 확률 `(C=t)`
  
      $$ y = \sigma (w^{T}+b) $$

      라고 할 때  
      (C=1)이면 y이고, (C=0)이면 1-y이므로,  
      아래와 같이 표현할 수 있다.

      $$p(C=t|x) = y^{t}(1-y)^{1-t} $$

  - 이 때 입력 데이터가 N개 존재하여, 그 값을 각각 <img src="/logistic_regression/tex/001fd25c5a52c6b4d381f8c071581bd5.svg?invert_in_darkmode&sanitize=true" align=middle width=108.04967414999999pt height=24.65753399999998pt/>이라고 하자.
  - 이 때 출력 데이터도 N개 존재하고, 그 값을 <img src="/logistic_regression/tex/ec9b770ea2cbdbac68a649eb61dc4a33.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06212004999999pt height=20.221802699999984pt/>이라고 하자.
  - N개의 입력 데이터에 대한 총 뉴런의 발화 확률은 아래와 같다.
   
  <p align="center"><img src="/logistic_regression/tex/fb87de8819758e1449cba05b722d31ee.svg?invert_in_darkmode&sanitize=true" align=middle width=511.09985520000004pt height=47.60747145pt/></p>

  - <img src="/logistic_regression/tex/e1693a187c655c73bb3552cfc9eb647b.svg?invert_in_darkmode&sanitize=true" align=middle width=50.54420084999999pt height=24.65753399999998pt/>를 **우도함수**라고 하며, 이 값이 최대가 되도록 파라미터를 수정하며 찾으면 네트워크가 학습이 잘 된 것이다.
 
  - 함수의 최대값을 찾을 때, 아래와 같이 특정 지점에서 기울기가 0이 되는 지점에 접근해가는 방법으로 찾는다.
   
  ![2](image/2.png)

  - 따라서, 위의 함수에서 기울기를 구하기 위해서는 위의 함수를 미분해야 하나, 함수의 곱의 형태를 미분하는 것은 복잡하기 때문에 아래와 같이 log를 취해서 위의 함수를 덧셈 형태로 변환한다.

    $$ E(w,b) = -\log{L(w,b)} = - \sum_{n=1}^{N} \{t_n\log{y_n} + (1-t_n)\log{(1-y_n)}\} $$

    위와 같은 형태의 함수를 **교차 엔트로피 오차 함수** 라고 한다.  
    줄여서 **오차 함수(error function)** 또는 **손실 함수(loss function)**라고 한다.

# 3.4.2.2 경사하강법

각 Parameter에 대한 기울기를 구하고, 그 기울기를 통해 오차를 수정해가면서 함수의 최대값을 찾아야 한다.  
따라서 파라미터에 대하여 다음 파라미터 값을 구하려면 그 파라미터에 대하여 편미분한 값을 찾아야 한다.  
아래는 오차 수정을 통한 다음 파라미터를 구하는 식이다.

<p align="center"><img src="/logistic_regression/tex/567cb0a3f8517ee1dd338bb87c787d15.svg?invert_in_darkmode&sanitize=true" align=middle width=190.8901005pt height=34.7253258pt/></p>
<p align="center"><img src="/logistic_regression/tex/651baaae2a01b0d0d6a333954300908e.svg?invert_in_darkmode&sanitize=true" align=middle width=180.57800145pt height=34.7253258pt/></p>

- 이 때, <img src="/logistic_regression/tex/2fd2435c6b8bb17613938c0b8628fc1e.svg?invert_in_darkmode&sanitize=true" align=middle width=22.27938899999999pt height=20.221802699999984pt/>는 학습률으로서 파라미터가 수렴되는 정도를 조절한다.
- 보통 0.1, 0.01 정도의 작은 값이 쓰인다.

- <img src="/logistic_regression/tex/f08ceabd708cda0932b98b9389561268.svg?invert_in_darkmode&sanitize=true" align=middle width=368.75387175pt height=32.256008400000006pt/>임을 기억한다.

<p align="center"><img src="/logistic_regression/tex/11e1f71886721f0e6e5c63c01f84032f.svg?invert_in_darkmode&sanitize=true" align=middle width=310.962333pt height=95.59347104999999pt/></p>

- 이와 같은 방법으로
<p align="center"><img src="/logistic_regression/tex/a55cb352d78e2e2f8f0be1c4e420f9b8.svg?invert_in_darkmode&sanitize=true" align=middle width=200.0449539pt height=47.60747145pt/></p>

- 따라서, 아래와 같이 다시 표현할 수 있다.

<p align="center"><img src="/logistic_regression/tex/48c2ac969354a65095228eba4020f161.svg?invert_in_darkmode&sanitize=true" align=middle width=226.8573285pt height=47.60747145pt/></p>
<p align="center"><img src="/logistic_regression/tex/afae1df7d79a2e5e433df1091f9a6bfc.svg?invert_in_darkmode&sanitize=true" align=middle width=199.02421934999998pt height=47.60747145pt/></p>