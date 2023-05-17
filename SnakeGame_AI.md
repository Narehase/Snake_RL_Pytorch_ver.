# Snake Game RL

## model
> model class는 deep Learning Network 구축을 도와줍니다.
> Parameter으로는 in_size, hi_size, out_size 가 있습니다.
> ``` python
>    def __init__(self, in_size, hi_size, out_size) -> None:
> ```

> __Parameter 설명 입니다.__
> ```python
> in_size:int #입력 받을 값의 Tensor Size 입니다. (None, in_size)
>
> hi_size:int #중간층인 hidden Layer의 Tensor Size 입니다. (None, hi_size)
>
> out_size:int #출력층의 Tensor Size 입니다. (None, out_size)
> ```

> 해당 class를 통해서 총 3개의 레이어를 지닌 network를 만들수 있게 됩니다.

``` python 

class model(nn.Module):
    def __init__(self, in_size, hi_size, out_size) -> None:
        super().__init__() #nn.Module 클래스 초기화

        self.Linear_L_Fast = nn.Linear(in_size, hi_size)
        self.Linear_L_Last = nn.Linear(hi_size, out_size)

    def forward(self, x):
        x = self.Linear_L_Fast(x)
        x = self.Linear_L_Last(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = 'D:\SnakeAI_model'
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),"./haha_E_gocsn_ne_model_eDa.pth")
```

## Q_learning
> Q_learning은 강화 학습 방법중에 하나입니다.
> 해당 class는 model의 학습을 도와 줍니다.<br>
> class의 Paremeter으로는 model, le, gamma 가 존재 합니다.
> ``` python 
>    def __init__(self, model:model, le:float, gamma:float) -> None:
> ```

> __Parameter 설명 입니다.__
> ```python
> model:model #위에서 설명한 model class입니다.
>
> le:float #learning rate을 설정 합니다.
>
> gamma:float #얼마나의 미래를 예측할지 Q_val 계산에 쓰이는 상수 입니다.
> ```

> __Q_value 구하는 공식은 다음과 같습니다.__ <br>
> $$ (GameOver = 1) (Qval = reward + gamma * max(State_{t+1}))$$
>$$ (GameOver = 0) (Qval = reward)$$
>위 식은 받은 리워드를 극대화해서 학습의 효율을 증가시킵니다.

> 해당 class를 가지고서 모델을 학습시킬수 있습니다.
``` python
class Q_learning:

    def __init__(self, model:model, le:float, gamma:float) -> None:
        self.model = model
        self.gamma = gamma
        self.le = le

        self.optimizer = optim.Adam(model.parameters())
        self.loss = nn.MSELoss()

    def tt(self, state, action, reward, nextStepState, done):
        # 여기는 numpy.array to torch.tensor convert
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        nextStepState = torch.tensor(nextStepState, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextStepState = torch.unsqueeze(nextStepState, 0)
            done = torch.unsqueeze(done, 0)

        pre = self.model(state)

        target = pre.clone()

        for index in range(len(done)):
            Q_value = reward[index]

            if not done[index]: # game over 시에
                Q_value = reward[index]+self.gamma*torch.max(self.model(nextStepState))
            target[index][torch.argmax(action)] = Q_value

        self.optimizer.zero_grad()
        loss = self.loss(target, pre)
        loss.backward()

        self.optimizer.step()
```
여기까지의 class 객체를 가지고 인공지능 학습을 할수 있게 됩니다.
___