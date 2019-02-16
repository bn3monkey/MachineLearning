# 5.1 기본사항

## 5.1.1 시계열 데이터

![5.1.1](image/1.PNG)

### 사인파 예측

![5.1.2](image/2.PNG)
출처 : https://stackoverflow.com/questions/1565115/approximating-function-with-neural-network?rq=1

## 5.1.2 과거의 은닉층

![5.1.3](image/3.PNG)

- 입력층 ->  은닉층

    입력 패러미터 : ![](https://latex.codecogs.com/svg.latex?x%28t%29%2c%2520h%28t%2d1%29)  출력 값 : ![](https://latex.codecogs.com/svg.latex?h%28t%29) 

    
![](https://latex.codecogs.com/svg.latex?h%28t%29%2520%3d%2520f%28Ux%28t%29%2520%2b%2520Wh%28t%2d1%29%2520%2b%2520b%29) 

- 은닉층 -> 출력층

    입력 패러미터 : ![](https://latex.codecogs.com/svg.latex?h%28t%29)  출력 값 : ![](https://latex.codecogs.com/svg.latex?y%28t%29) 

    
![](https://latex.codecogs.com/svg.latex?%2520y%28t%29%2520%2520%3d%2520g%28Vh%28t%29%2520%2b%2520c%29%2520) 

- 오차 함수

     ![](https://latex.codecogs.com/svg.latex?%2520E%2520%3d%2520E%28U%2cV%2cW%2cb%2cc%29%2520)   

    이 떄, ![](https://latex.codecogs.com/svg.latex?p%28t%29%2520%3d%2520%2520Ux%28t%29%2520%2b%2520Wh%28t%2d1%29%2520%2b%2520b%2c%2520q%28t%29%2520%3d%2520Vh%28t%29%2520%2b%2520c) 라고 하면,  
    ![](https://latex.codecogs.com/svg.latex?h%28t%29%2520%3d%2520f%28p%28t%29)  , ![](https://latex.codecogs.com/svg.latex?y%28t%29%2520%3d%2520g%28q%28t%29%29) 이다.

    따라서, 오차함수에서 각각 은닉층에 따른 오차항과 출력층에 따른 오차항을  
    ![](https://latex.codecogs.com/svg.latex?e%5f%7bh%7d%28t%29) , ![](https://latex.codecogs.com/svg.latex?e%5f%7bo%7d%28t%29) 라고 할때, 아래와 같이 쓸 수 있다.

    
![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%29%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520p%28t%29%7d%7d%2520%2520) 
    
![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bo%7d%28t%29%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520q%28t%29%7d%7d%2520%2520) 

    오차 함수의 패러미터인 ![](https://latex.codecogs.com/svg.latex?U) , ![](https://latex.codecogs.com/svg.latex?V) , ![](https://latex.codecogs.com/svg.latex?W) , ![](https://latex.codecogs.com/svg.latex?b) , ![](https://latex.codecogs.com/svg.latex?c) 에 대한 오차항은 아래와 같다.

    
![](https://latex.codecogs.com/svg.latex?%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520U%7d%7d%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520p%28t%29%7d%7d%28%7b%7b%5cdelta%2520p%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520U%7d%7d%29%5e%7bT%7d%2520%3d%2520e%5f%7bh%7d%28t%29x%28t%29%5e%7bT%7d) 

    
![](https://latex.codecogs.com/svg.latex?%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520V%7d%7d%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520q%28t%29%7d%7d%28%7b%7b%5cdelta%2520q%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520V%7d%7d%29%5e%7bT%7d%2520%3d%2520e%5f%7bo%7d%28t%29h%28t%29%5e%7bT%7d) 

    
![](https://latex.codecogs.com/svg.latex?%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520W%7d%7d%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520p%28t%29%7d%7d%28%7b%7b%5cdelta%2520p%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520W%7d%7d%29%5e%7bT%7d%2520%3d%2520e%5f%7bo%7d%28t%29h%28t%2d1%29%5e%7bT%7d) 
    
    
![](https://latex.codecogs.com/svg.latex?%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520b%7d%7d%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520p%28t%29%7d%7d%28%7b%7b%5cdelta%2520p%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520b%7d%7d%29%2520%3d%2520e%5f%7bh%7d%28t%29%2520) 

    
![](https://latex.codecogs.com/svg.latex?%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520c%7d%7d%2520%3d%2520%7b%7b%5cdelta%2520E%7d%2520%5cover%2520%7b%5cdelta%2520q%28t%29%7d%7d%28%7b%7b%5cdelta%2520q%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520c%7d%7d%29%2520%3d%2520e%5f%7bo%7d%28t%29) 

- 사인파 모델에서의 출력 함수.

   - 기존의 모델(소프트맥스 함수, 시그모이드 함수)에서는 y(t)의 값이 0 혹은 1을 가지고 있는 이산값이나, 0.0 ~ 1.0 내의 확률이었음.
    
![](https://latex.codecogs.com/svg.latex?%2520y%28t%29%2520%2520%3d%2520g%28Vh%28t%29%2520%2b%2520c%29%2520) 

   - 사인파의 경우, 출력값이 확률이 아니라 함수값이어야 하므로 그대로 사용
    
![](https://latex.codecogs.com/svg.latex?%2520y%28t%29%2520%3d%2520Vh%28t%29%2520%2b%2520c%2520%2c%2520g%28x%29%2520%3d%2520x%2520) 


- 사인파 모델에서 오차 함수

  - 모델의 예측값 ![](https://latex.codecogs.com/svg.latex?y%28t%29) 와 정답인 값 ![](https://latex.codecogs.com/svg.latex?t%28t%29) 와의 오차를 최소화해야 함.
  - 아래와 같은 제곱 오차함수를 통해 0~T까지의 모든 예측값과 정답값의 오차의 절대값이 합이 최소가 되도록 오차함수를  만든다.

    
![](https://latex.codecogs.com/svg.latex?%2520E%2520%3d%2520%7b%7b1%7d%2520%5cover%2520%7b2%7d%7d%5csum%5e%7bT%7d%5f%7bt%3d1%7d%7c%7cy%28t%29%2dt%28t%29%7c%7c%5e%7b2%7d%2520) 

## 5.1.3 Backpropagation Through  Time

- 위의 제곱 오차함수를 적용할 때, 오차항은 아래와 같다.


![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%29%2520%3d%2520f%27%28p%28t%29%29%2520V%5e%7bT%7d%2520e%5f%7b0%7d%28t%29%2520%2520) 

![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bo%7d%28t%29%2520%3d%2520g%27%28q%28t%29%29%2520%28y%28t%29%2520%2520%2d%2520t%28t%29%29%2520%2520) 

- 은닉층 h(t)는 h(t-1)의 영향을 받는다. 따라서, 역전파시 t-1에 대한 오차 역시 생각해주어야 한다.

- 오차를 시간을 거슬러 역전파하게 되는 것을 Backpropagation through time, 줄여서 BPTT라고 한다.

![5.1.4](image/4.PNG)

- 은닉층 h(t)는 h(t-1)의 영향을 받으므로, h(t)의 오차인 ![](https://latex.codecogs.com/svg.latex?e%5f%7bh%7d%28t%29) 에 대한 식으로 h(t-1)의 오차인 ![](https://latex.codecogs.com/svg.latex?e%5f%7bh%7d%28t%2d1%29) 를 표현해야 한다.


![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%2d1%29%2520%3d%2520%7b%2520%7b%5cdelta%2520E%7d%5cover%7b%5cdelta%2520p%28t%29%7d%2520%7d%2520%7b%7b%5cdelta%2520p%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520p%28t%2d1%29%7d%7d%2520) 


![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%2d1%29%2520%3d%2520e%5fh%28t%29%2520%7b%7b%5cdelta%2520p%28t%29%7d%2520%5cover%2520%7b%5cdelta%2520h%28t%2d1%29%7d%7d%2520%7b%7b%5cdelta%2520h%28t%2d1%29%7d%5cover%7b%5cdelta%2520p%28t%2d1%29%7d%7d%2520) 


![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%2d1%29%2520%3d%2520e%5fh%28t%29%2520%28Wf%27%28p%28t%2d1%29%29%29%2520) 

- 이를 재귀적으로 내려가면, 은닉층 h(t-z)와 h(t-z-1)에서의 오차 ![](https://latex.codecogs.com/svg.latex?e%5f%7bh%7d%28t%2dz%29) , ![](https://latex.codecogs.com/svg.latex?e%5f%7bh%7d%28t%2dz%2d1%29) 에 대해서 구할 수 있다.


![](https://latex.codecogs.com/svg.latex?%2520e%5f%7bh%7d%28t%2dz%2d1%29%2520%3d%2520e%5fh%28t%2dz%29%2520%28Wf%27%28p%28t%2dz%2d1%29%29%29%2520) 

- 이 오차를 모든 매개변수에 대해 나타내면 아래와 같다.

![5.1.4](image/5.PNG)

- 이 때, ![](https://latex.codecogs.com/svg.latex?%5cgamma) 는 얼마만큼 과거로 올라갈 건지에 대해 나타내는 변수이다. 결과가 소실되는 것을 방지하기 위해 10~100 정도로 잡는다.