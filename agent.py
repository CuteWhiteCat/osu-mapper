import numpy as np
from mapper_env import MapperEnv
import matplotlib.pyplot as plt

# 建立環境
env = MapperEnv(audio="audios/audio.mp3")

# 設定參數空間的每個離散值
discrete_values = [
    [2.0, 3.0, 4.0, 5.0, 6.0],                                                        # circle_size
    [3.0, 3.5, 4.0, 4.5 ,5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],     # overall_difficulty
    [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],                         # approach_rate
    [0.8, 1.1, 1.4, 1.7, 2.0],                                                        # slider_multiplier
    [0.6, 0.8, 1.0, 1.2, 1.4],                                                        # temperature
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],                              # cfg_scale
    [0, 1],                                                                           # super_timing
    [0],                                                                              # mapper_id
    [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],               # year
]

# Q-Table 以 tuple(action) 為 key
q_table = {}

# 訓練參數
episodes = 10
alpha = 0.1
gamma = 0.95
epsilon = 0.2

rewards = []

for episode in range(episodes):
    # ε-greedy 策略選擇動作
    if np.random.rand() < epsilon or len(q_table) == 0:
        # 隨機組合一組 action
        action = [np.random.choice(vs) for vs in discrete_values]
    else:
        # 選擇最大 Q 值的 action（tuple 形式）
        best_action_tuple = max(q_table.items(), key=lambda item: item[1])[0]
        action = list(best_action_tuple)

    action_array = np.array(action, dtype=np.float32)
    action_key = tuple(np.round(action_array, 3))  # 避免浮點誤差

    # 執行環境 step
    reward = env.step(action_array)

    # 初始化 Q 值
    if action_key not in q_table:
        q_table[action_key] = 0.0

    # Q-Learning 更新
    max_next_q = max(q_table.values(), default=0)
    q_table[action_key] += alpha * (reward + gamma * max_next_q - q_table[action_key])

    rewards.append(reward)
    print(f"Action: {action_key}")
    print(f"Episode {episode+1}/{episodes}: Reward = {reward:.4f}")

plt.plot(rewards, label='reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.savefig("reward_curve.png")
plt.close()

# 顯示最佳結果
best_action = max(q_table.items(), key=lambda item: item[1])
print("\n✅ 最佳參數組合：")
print(f"Action: {best_action[0]}")
print(f"Estimated Q-value: {best_action[1]:.4f}")