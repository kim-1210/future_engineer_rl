from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class FutureEngineerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 한 번 학습 업데이트 전에 각 환경에서 모을 step 수
    num_steps_per_env = 24

    # 전체 학습 반복 횟수
    max_iterations = 15000

    # 몇 iteration마다 모델을 저장할지
    save_interval = 50

    # logs/rsl_rl 아래에 만들어질 실험 폴더 이름
    experiment_name = "future_engineer_task"

    # 같은 실험 안에서 run 이름을 따로 붙이고 싶을 때 사용
    run_name = ""

    # policy 출력 action을 이 범위로 자르고 싶을 때 사용
    # 현재 action 코드에서 [-1, 1]로 한 번 더 clamp하므로 보통 꺼둬도 됨
    # clip_actions = 10.0

    # actor는 행동을 고르는 네트워크, critic은 현재 상태의 가치를 평가하는 네트워크
    policy = RslRlPpoActorCriticCfg(
        # 학습 시작 시 action을 얼마나 랜덤하게 뽑을지 정하는 초기 표준편차
        init_noise_std=0.7,

        # actor 입력 관측값을 자동 정규화할지 여부
        actor_obs_normalization=False,

        # critic 입력 관측값을 자동 정규화할지 여부
        critic_obs_normalization=False,

        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],

        # 네트워크 활성화 함수. ELU = RL에서 안정적일 때가 있음
        activation="elu",
    )

    # PPO 알고리즘의 학습 규칙 설정
    algorithm = RslRlPpoAlgorithmCfg(
        # critic(value) 오차를 loss에 얼마나 크게 반영할지
        value_loss_coef=1.0,

        # value 업데이트도 너무 크게 바뀌지 않게 제한할지
        use_clipped_value_loss=True,

        # policy가 한 번에 너무 많이 바뀌지 못하게 막는 PPO 제한 폭
        clip_param=0.2,

        # 탐색을 얼마나 장려할지. 크면 action std가 너무 빨리 줄어드는 것을 막는 힘이 커짐
        entropy_coef=0.002,

        # 모은 데이터를 몇 번 반복해서 학습할지
        num_learning_epochs=5,

        # 모은 데이터를 몇 조각으로 나누어 mini-batch 학습할지
        num_mini_batches=4,

        # 한 번 업데이트할 때 weight를 얼마나 크게 움직일지
        learning_rate=1e-4,

        # adaptive는 이 작업에서 lr을 0.01까지 키워 정책을 망가뜨렸다.
        # 고정 lr로 두어 좋아진 행동을 한 번에 밀어버리지 않게 한다.
        schedule="fixed",

        # 미래 보상을 현재 가치로 얼마나 반영할지. 1에 가까울수록 먼 미래도 중요하게 봄
        gamma=0.99,

        # advantage 계산을 얼마나 부드럽게 할지 정하는 GAE 계수
        lam=0.95,

        # adaptive schedule이 목표로 삼는 policy 변화량
        desired_kl=0.01,

        # gradient가 너무 커져서 학습이 튀는 것을 막는 최대 크기
        max_grad_norm=1.0,
    )
