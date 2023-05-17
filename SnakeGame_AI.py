import numpy as np
from collections import UserDict, deque, namedtuple
import msvcrt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import os

Max_Size = 100


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LE = 0.001

class AI_User: # Agent Clss produce
    def __init__(self):
        self.n_game = 0 
        # 현재의 게임 반복 학습
        self.total_Score = 0
        # 여태까지 얻은 게임 점수
        self.entropy = 10
        # 무작위로 활동힐 확룰
        self.gamma = 0.95
        # 얼마나의 미래를 예측할지 에 대한 보정 값 - Q_value 계산

        self.memoryss = deque(maxlen=MAX_MEMORY) #deque << 이거 신기함
        # 이전에 경험한 데이터의 저장 량 제한
        self.model = model(13,512,4) 
        # nn모델 생성
        self.Qr = Q_learning(self.model,le=LE,gamma=self.gamma)
        # 총합 합성모델 
    
    def memorys(self,state,action,reward,next_state,done):
        self.memoryss.append((state,action,reward,next_state,done))
        # 내 마음속에 저장 ~ *

    def shot_run(self ,state ,action ,reward ,next_state ,done):
        self.Qr.tt(state ,action ,reward ,next_state ,done)
        # 짧게 공부하자! 

    def long_run(self):
        if(len(self.memoryss) - 1) > BATCH_SIZE:
            
            mini = random.sample(self.memoryss, BATCH_SIZE)
        else: mini = self.memoryss

        state_s,action_s,reward_s,next_state_s,done_s = zip(*mini)
        self.Qr.tt(state_s,action_s,reward_s,next_state_s,done_s)
        # 기합 넣고 기억하자--!!

    def Get_Action(self, state):
        
        final_move = [0,0,0,0]
        if(np.random.randint(0,200)<self.entropy and self.n_game < 150):
            move = np.random.randint(0,4)
            final_move[move]=1
        else:
            state = torch.tensor(state,dtype=torch.float)
            pre = self.model(state)
            move = torch.argmax(pre).item()
            final_move[move]=1 
        return final_move
        # 어리로 가야하오.

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
        torch.save(self.state_dict(),"./model.pth")
        # 메모메모 저장 하자!
    
class Q_learning:

    def __init__(self, model:model, le:float, gamma:float) -> None:
        self.model = model
        self.gamma = gamma
        self.le = le

        self.optimizer = optim.Adam(model.parameters())
        self.loss = nn.MSELoss()

    def tt(self, state, action, reward, nextStepState, done):
        # 여기서 부터 numpy.array to torch.tensor convert
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

        # 여기까지!!!

        # print(state)
        # print(action.shape, "&&")
        pre = self.model(state)
        # print(pre.shape, "&&")
        target = pre.clone()
        # print(target.shape, "&&")

        for index in range(len(done)):
            Q_value = reward[index]

            if not done[index]: # game over
                Q_value = reward[index]+self.gamma*torch.max(self.model(nextStepState))
            target[index][torch.argmax(action)] = Q_value

        self.optimizer.zero_grad()
        loss = self.loss(target, pre)
        loss.backward()

        self.optimizer.step()
    # 공부는 이순신 공부법! 아직 신에게는 12시간의 시간이 남았습니다.
class df:
    DOWN = 0
    RIGHT = 90
    UP = 180
    LEFT = 270

class Game:
    def __init__(self) -> None:
        
        self.FildPointType = namedtuple('Fild', 'x, y')
        self.FildPoint = self.FildPointType(10, 10)

        
        self.HeadPointType = namedtuple('HeadPoint', 'x, y')
        self.HeadPoint = self.HeadPointType(5, 5)
        self.Re_ = 0

        a,b = self.getApple()
        self.ApplePointType = namedtuple('apple', 'x, y')
        self.ApplePoint = self.ApplePointType(a, b)

        self.Body_Memory = deque(maxlen= Max_Size)

        self.len = 5
        self.deg = 0

    def Move(self):
        deg = self.deg
    
        ex = int(np.sin(np.radians(deg))) + self.HeadPoint.x
        ey = int(np.cos(np.radians(deg))) + self.HeadPoint.y
        
        assi = np.array([ey, ex])

        apic = np.min(np.array([self.ApplePoint.x == ex, self.ApplePoint.y == ey]))
        print(apic)
        if apic:
            a,b = self.getApple()
            self.ApplePoint = self.ApplePointType(a,b)
            self.len += 1
            
        ac = False
        # print(self.Body_Memory)
        # for i in self.Body_Memory:
        #     if i.y == self.HeadPoint.y and i.x == self.HeadPoint.x:
        #         ac = True
        #         break

        self.Body_Memory.appendleft(self.HeadPoint)
        if self.len < len(self.Body_Memory):
            self.Body_Memory.pop()

        if (np.min(assi) < 0) or (np.max(assi) >= self.FildPoint.x) or ac:
            self.Re_ = 0
            self.HeadPoint = self.HeadPointType(ex,ey)
            return False, -1
        else:
            
            xDit = self.HeadPoint.x - self.ApplePoint.x
            yDit = self.HeadPoint.y - self.ApplePoint.y
            self.Re_ = np.sqrt(abs(yDit**2 + xDit**2))
            self.HeadPoint = self.HeadPointType(ex,ey)
            return True, apic

    def getApple(self):
        x, y = np.random.randint(0, self.FildPoint.x, (2))
        return y, x        
    
    def get_State(self):
        na = np.zeros(4)
        for i in range(4):
            iss = i* 90    
            ex = int(np.sin(np.radians(self.deg+iss))) + self.HeadPoint.x
            ey = int(np.cos(np.radians(self.deg+iss))) + self.HeadPoint.y
            if (ex < 0 or ex >= self.FildPoint.x) and\
                (ey < 0 or ey >= self.FildPoint.x): 
                na[i] =  False
            else:
                na[i] = True
                for fep in self.Body_Memory:
                    if fep.x == self.HeadPoint.x and\
                    fep.y == self.HeadPoint.y :
                        na[i] = False
                    
        a,b,d,c = na
        
        xDit = self.HeadPoint.x - self.ApplePoint.x
        yDit = self.HeadPoint.y - self.ApplePoint.y
        Home = np.sqrt(abs(yDit**2 + xDit**2))

        diu = self.deg == df.UP
        did = self.deg == df.DOWN
        dil = self.deg == df.LEFT
        dir = self.deg == df.RIGHT

        state = [
            a,b,d,c,

            
            diu,
            did,
            dir,
            dil,

            self.ApplePoint.x < self.HeadPoint.x, #절대적 지표
            self.ApplePoint.x > self.HeadPoint.x,
            self.ApplePoint.y < self.HeadPoint.y,
            self.ApplePoint.y > self.HeadPoint.y,
    
            self.Re_ > Home
        ]

        return np.array(state, dtype = np.int32)
    

    def Display(self):
        self.Fild = np.zeros([self.FildPoint.y, self.FildPoint.x])

        
        self.Fild[self.HeadPoint.y, self.HeadPoint.x] = 1
        for i in self.Body_Memory:
            self.Fild[i.y, i.x] = 0.5
        self.Fild[self.ApplePoint.y,self.ApplePoint.x] = 3


        pizz = ("#" * (self.Fild.shape[0]+2 ))+ "\n#"
        for y in self.Fild:
            for x in y:
                a = " "
                # a = "□"
                if x == 0:
                    a = " "
                elif x > 0 and x < 1:
                    a = "■"
                elif x == 3:
                    a = "$"
                elif x == 1:
                    a = "▤"
                else:
                    a = str(x)
                pizz+= a
            pizz+= "#\n#"
        pizz += ("#" * (self.Fild.shape[0]+1 ))+"\n"
        # print("\033[H\033[J")
        print(f"\033[{0};{0}H",end="")
        print(pizz)

if __name__ in "__main__":
    os.system("cls")
    a = Game()
    user = AI_User()
    total_score = 0
    record = 0
    push = 0.0
    lu = 0

    while True:
        if user.n_game > 100:
            a.Display()
            time.sleep(0.05)
        else:
            print(f"\033[{0};{0}H",end="")
            a.Display()
            # time.sleep(0.1)

        oldState = a.get_State()
        print(oldState)
        Move = user.Get_Action(oldState)

        a.deg = np.argmax(np.array(Move))*90
        _, re = a.Move()
        re+= push*(re>0)
        score = a.len - 5
        news = a.get_State()

        user.shot_run(oldState, Move, re, news,_)
        user.memorys(oldState, Move, re, news,_)

        lu+=1 # 반복 행동 막기

        if lu > 80 and re != 1:
            _ = False
        elif re > 0:
            push += 0.2
            lu = 0
        print('Game:',user.n_game,'\tScore:',score,'\tRecord:',record,'\tTotal_Score:',total_score,"\t","lu : ", lu, "\t\r")
        if not _:
            push = 0.
            a = Game()
            user.n_game += 1
            user.long_run()
            if(score > re):
                re = score
                user.model.save()
            if total_score < score:
                total_score = score
            # print('Game:',user.n_game,'\tScore:',score,'\tRecord:',record,'\tTotal_Score:',total_score,"\t","lu : ", lu, "\t")
            lu = 0
            
        if user.n_game % 50 == 0:
            os.system("cls")
