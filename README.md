# 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/201888ae-bdfb-4555-9388-9e9bae45b15f"" width="30%" height="30%"></p>

본 레포지토리는 Continuous Control Task를 위한 강화학습 알고리즘인 **Twin Delayed Deep Deterministic Policy Gradient (TD3)**의 PyTorch 구현을 담고있습니다.
TD3는 Actor-Critic 방법론에서 발생하는 overestimation bias 문제를 해결하기 위해 Deep Deterministic Policy Gradient (DDPG)를 기반으로 세 가지 핵심 개선사항을 도입한 알고리즘입니다.
레포지토리의 목적은 2025년 하반기 서강대학교 AI·SW 대학원의 "강화학습의 기초" 과목의 Term Project 코드 산출물의 저장 및 실행 방법에 대한 설명을 포함하기 위함입니다.

# 2. 레포지토리 구조
```
td3-implementation/
├── results                # 학습된 모델을 저장하기 위한 폴더
├── trained_model          # 실험을 수행하며 학습된 각 환경에 대한 actor, critic 모델 파일이 있음 (.pth)
├── buffer.py              # replay buffer python 구현
├── train.py               # TD3 알고리즘의 학습 루프
├── evaluation.py          # 평가 프로토콜 및 메트릭
├── network.py             # actor, critic 딥러닝 네트워크의 pytorch 구현
└── requirements.txt       # 레포지토리 실행을 위한 라이브러리 이름 및 버전 명시
```

# 3. 모델 학습
TD3 알고리즘을 학습하기 위해서 train.py를 실행해야 합니다. train.py 코드의 전체 구조는 아래와 같습니다.

- 하이퍼파라미터 설정(argparse) : 환경, 시드, 학습 스텝 수, 노이즈 파라미터 등을 명령줄에서 설정
- 평가 함수 (eval_policy) : 학습된 policy를 10개 episode에서 테스트하여 평균 return 계산
- 초기화 : Gym 환경 생성 및 랜덤 시드 설정 -> Actor/Critic 네트워크와 TD3 policy 객체 생성 -> Replay 버퍼 생성
- 학습 루프 : 총 1,000,000 스텝에 대하여 TD3 모델 학습

## 3.1 하이퍼파라미터 설명
```
--env: 학습할 OpenAI Gym 환경
--seed: 재현성을 위한 랜덤 시드
--start_timesteps: 초기 random exploration 스텝 수, 기본 값 10000
--eval_frequency: 평가 주기, 기본 값 5000
--max_timesteps: 총 학습 스텝 수, 기본 값 1e6
--exploration_noise: exploration을 위한 gaussian noise std., 기본 값 0.1
--batch_size: mini-batch size, 기본 값 256
--discount: discount factor gamma, 기본 값 0.99
--tau: target newtork soft update 비율, 기본 값 0.005
--policy_noise: target policy smoothing noise, 기본 값 0.2
--noise_clip: target policy noise clipping 범위, 기본 값 0.5
--policy_frequency: policy delayed update frequency, 기본 값 2
--model_save_path: 모델을 저장할 폴더 및 모델명 입력
--model_load_path: transfer learning을 수행할 학습된 모델 경로 입력
```

## 3.2 모델 학습
cmd에서 아래와 같은 명령어를 입력하여 모델을 학습할 수 있습니다.

```
python train.py --env "HalfCheetah-v4" --model_save_path "./results/01_HalfCheetah/01_TD3_Best_Params.pth"
```
3.1에 정의된 하이퍼파라미터를 변경하면서 학습을 시도해볼 수 있습니다. 학습을 실행하면 1,000 step 단위로 evaluation을 수행하고 해당 episode에 대한 average return 값을 실시간으로 확인할 수 있습니다.
또한, eval_frequency step마다 actor, critic 모델 및 append된 total return값의 리스트가 model_save_path 경로에 저장됩니다. 학습이 cmd에서 실행되고 있는 모습을 아래 그림과 같이 첨부합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/dcccaf78-0734-4348-a812-088f03ceab1e" width="40%" height="40%"></p>

# 4. Evaluation
TD3 알고리즘을 평가하기 위해서 evaluation.py를 실행해야 합니다. evaluation.py 코드의 전체 구조는 아래와 같습니다.

- 하이퍼파라미터 설정(argparse) : 환경, 시드, 렌더링 모드, 학습된 모델 경로 설정
- 평가 함수 (eval_policy) : 하나의 episode를 실행하고 total return 반환 -> 랜덤 시드로 환경을 초기화하여 재현성 확보
- 모델 로딩 : 환경 생성 및 state/action 차원 호출 -> Actor 네트워크에 학습된 가중치 로드 -> TD3 policy 객체 생성
- 다중 시드 평가 : 학습된 모델을 10개의 다른 랜덤 시드에서 테스트하여 일반화 성능을 평가함

## 4.1 하이퍼파라미터 설명
```
--env: 학습할 OpenAI Gym 환경
--seed: 재현성을 위한 랜덤 시드
--render_mode: 모델과 환경의 상호작용을 시각적으로 확인할 수 있는 모드, 기본 값 None 또는 "human"
--trained_actor_path: 평가 수행시, actor만 필요하므로 학습된 actor 모델의 경로 입력
```

## 4.2 모델 평가
cmd에서 아래와 같은 명령어를 입력하여 모델을 평가할 수 있습니다.

```
python evaluation.py --env "HalfCheetah-v4" --trained_actor_path "./trained_model/01_HalfCheetah/TD3_Ours_actor.pth" --render_mode "human"
```

평가를 실행하면, 학습된 모델을 10개의 다른 랜덤 시드에서 테스트하여 일반화 성능을 평가합니다. HalfCheetah 환경에서 baseline actor 모델의 평가 결과를 아래 그림과 같이 첨부합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/b8883551-ff30-48c3-b414-0c30fa8fdbc8" width="40%" height="40%"></p>

# 5. 환경별 최고 성능

<p align="left"><img src="https://github.com/user-attachments/assets/d1758841-187d-48fa-b7c8-a76d6aa9a89d" width="70%" height="70%"></p>

- 학습 완료 후, 10개의 random seed를 사용하여 각 episode에 대한 return의 평균 및 표준편차를 기록
- baseline : 논문에서 제시한 파라미터의 기본 값을 이용한 학습 결과를 기록
- ours : 나머지 하이퍼파라미터 중 평균 return이 가장 높은 학습 결과를 기록
