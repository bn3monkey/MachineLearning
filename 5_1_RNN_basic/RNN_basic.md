# 5.1 기본사항

## 5.1.1 시계열 데이터

![5.1.1](image/1.PNG)

### 사인파 예측

![5.1.2](image/2.PNG)
출처 : https://stackoverflow.com/questions/1565115/approximating-function-with-neural-network?rq=1

## 5.1.2 과거의 은닉층

![5.1.3](image/3.PNG)

- 입력층 ->  은닉층

    입력 패러미터 :  <script type="math/tex; mode=display" id="MathJax-Element-1">x(t), h(t-1)</script>출력 값 : 
<script type="math/tex; mode=display" id="MathJax-Element-1">h(t)</script>
    
<script type="math/tex; mode=display" id="MathJax-Element-1">h(t) = f(Ux(t) + Wh(t-1) + b)</script>
- 은닉층 -> 출력층

    입력 패러미터 :  <script type="math/tex; mode=display" id="MathJax-Element-1">h(t)</script>출력 값 : 
<script type="math/tex; mode=display" id="MathJax-Element-1">y(t)</script>
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> y(t)  = g(Vh(t) + c) </script>
- 오차 함수

     <script type="math/tex; mode=display" id="MathJax-Element-1"> E = E(U,V,W,b,c) </script> 

    이 떄, 라<script type="math/tex; mode=display" id="MathJax-Element-1">p(t) =  Ux(t) + Wh(t-1) + b, q(t) = Vh(t) + c</script>고 하면,  
     <script type="math/tex; mode=display" id="MathJax-Element-1">h(t) = f(p(t)</script>, 이<script type="math/tex; mode=display" id="MathJax-Element-1">y(t) = g(q(t))</script>다.

    따라서, 오차함수에서 각각 은닉층에 따른 오차항과 출력층에 따른 오차항을  
    ,<script type="math/tex; mode=display" id="MathJax-Element-1">e_{h}(t)</script> 라<script type="math/tex; mode=display" id="MathJax-Element-1">e_{o}(t)</script>고 할때, 아래와 같이 쓸 수 있다.

    
<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t) = {{\delta E} \over {\delta p(t)}}  </script>    
<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{o}(t) = {{\delta E} \over {\delta q(t)}}  </script>
    오차 함수의 패러미터인 ,<script type="math/tex; mode=display" id="MathJax-Element-1">U</script> ,<script type="math/tex; mode=display" id="MathJax-Element-1">V</script> ,<script type="math/tex; mode=display" id="MathJax-Element-1">W</script> ,<script type="math/tex; mode=display" id="MathJax-Element-1">b</script> 에<script type="math/tex; mode=display" id="MathJax-Element-1">c</script> 대한 오차항은 아래와 같다.

    
<script type="math/tex; mode=display" id="MathJax-Element-1"> {{\delta E} \over {\delta U}} = {{\delta E} \over {\delta p(t)}}({{\delta p(t)} \over {\delta U}})^{T} = e_{h}(t)x(t)^{T}</script>
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> {{\delta E} \over {\delta V}} = {{\delta E} \over {\delta q(t)}}({{\delta q(t)} \over {\delta V}})^{T} = e_{o}(t)h(t)^{T}</script>
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> {{\delta E} \over {\delta W}} = {{\delta E} \over {\delta p(t)}}({{\delta p(t)} \over {\delta W}})^{T} = e_{o}(t)h(t-1)^{T}</script>    
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> {{\delta E} \over {\delta b}} = {{\delta E} \over {\delta p(t)}}({{\delta p(t)} \over {\delta b}}) = e_{h}(t) </script>
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> {{\delta E} \over {\delta c}} = {{\delta E} \over {\delta q(t)}}({{\delta q(t)} \over {\delta c}}) = e_{o}(t)</script>
- 사인파 모델에서의 출력 함수.

   - 기존의 모델(소프트맥스 함수, 시그모이드 함수)에서는 y(t)의 값이 0 혹은 1을 가지고 있는 이산값이나, 0.0 ~ 1.0 내의 확률이었음.
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> y(t)  = g(Vh(t) + c) </script>
   - 사인파의 경우, 출력값이 확률이 아니라 함수값이어야 하므로 그대로 사용
    
<script type="math/tex; mode=display" id="MathJax-Element-1"> y(t) = Vh(t) + c , g(x) = x </script>

- 사인파 모델에서 오차 함수

  - 모델의 예측값 와<script type="math/tex; mode=display" id="MathJax-Element-1">y(t)</script> 정답인 값 와<script type="math/tex; mode=display" id="MathJax-Element-1">t(t)</script>의 오차를 최소화해야 함.
  - 아래와 같은 제곱 오차함수를 통해 0~T까지의 모든 예측값과 정답값의 오차의 절대값이 합이 최소가 되도록 오차함수를  만든다.

    
<script type="math/tex; mode=display" id="MathJax-Element-1"> E = {{1} \over {2}}\sum^{T}_{t=1}||y(t)-t(t)||^{2} </script>
## 5.1.3 Backpropagation Through  Time

- 위의 제곱 오차함수를 적용할 때, 오차항은 아래와 같다.


<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t) = f'(p(t)) V^{T} e_{0}(t)  </script>
<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{o}(t) = g'(q(t)) (y(t)  - t(t))  </script>
- 은닉층 h(t)는 h(t-1)의 영향을 받는다. 따라서, 역전파시 t-1에 대한 오차 역시 생각해주어야 한다.

- 오차를 시간을 거슬러 역전파하게 되는 것을 Backpropagation through time, 줄여서 BPTT라고 한다.

![5.1.4](image/4.PNG)

- 은닉층 h(t)는 h(t-1)의 영향을 받으므로, h(t)의 오차인 에<script type="math/tex; mode=display" id="MathJax-Element-1">e_{h}(t)</script> 대한 식으로 h(t-1)의 오차인 를<script type="math/tex; mode=display" id="MathJax-Element-1">e_{h}(t-1)</script> 표현해야 한다.


<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t-1) = { {\delta E}\over{\delta p(t)} } {{\delta p(t)} \over {\delta p(t-1)}} </script>

<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t-1) = e_h(t) {{\delta p(t)} \over {\delta h(t-1)}} {{\delta h(t-1)}\over{\delta p(t-1)}} </script>

<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t-1) = e_h(t) (Wf'(p(t-1))) </script>
- 이를 재귀적으로 내려가면, 은닉층 h(t-z)와 h(t-z-1)에서의 오차 ,<script type="math/tex; mode=display" id="MathJax-Element-1">e_{h}(t-z)</script> 에<script type="math/tex; mode=display" id="MathJax-Element-1">e_{h}(t-z-1)</script> 대해서 구할 수 있다.


<script type="math/tex; mode=display" id="MathJax-Element-1"> e_{h}(t-z-1) = e_h(t-z) (Wf'(p(t-z-1))) </script>
- 이 오차를 모든 매개변수에 대해 나타내면 아래와 같다.

![5.1.4](image/5.PNG)

- 이 때, 는<script type="math/tex; mode=display" id="MathJax-Element-1">\gamma</script> 얼마만큼 과거로 올라갈 건지에 대해 나타내는 변수이다. 결과가 소실되는 것을 방지하기 위해 10~100 정도로 잡는다.