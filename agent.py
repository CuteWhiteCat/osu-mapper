import numpy as np
import random
from itertools import combinations
from mapper_env import MapperEnv
import matplotlib.pyplot as plt

# 建立環境
env = MapperEnv(audio="audios/audio.mp3")

# 設定參數空間的每個離散值
discrete_values = [
    [3.5, 4.0, 4.2, 4.5, 4.8, 5.0, 5.5, 6.0],
    [7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    [8.0, 8.5, 8.7, 9.0, 9.3, 9.5, 9.8, 10.0],
    [0.8, 1.0, 1.2, 1.4, 1.7, 2.0],
    [0.6, 0.8, 1.0, 1.2, 1.4],
    [1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [0, 1],
    [6381153, 8688812, 3250792, 1634445, 4219034, 2202163, 1661227, 199669, 6443117, 896613, 26232141, 12763959, 2506267, 10354460, 9088487, 8095457, 11758667, 7354729, 365586, 4413842, 8953955, 1623405, 9014033, 3334163, 4378277, 2706438, 4452992, 9426712, 3620525, 4783121, 9320502, 3638962, 4323406, 1474421, 266596, 416662, 3621552, 1721120, 15959751, 4754771, 3552948, 5404892, 11545816, 14729352, 7715620, 5115995, 2542623, 14261540, 9416836, 873961, 1047883, 1848318, 13400075, 1947052, 1826598, 7451883, 10284894, 16080485, 3664366, 10631018, 15822813, 2683648, 1787171, 4944043, 3475189, 2688581, 11443437, 7733431, 3723568, 870584, 1872496, 1592447, 6203841, 873758, 53378, 4785223, 12914206, 14305018, 243917, 50265, 2123087, 1715720, 16368250, 1171628, 284905, 1343783, 4541873, 4086497, 44308, 1249323],
    [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
]

descriptors = [
    "messy", "geometric", "grid snap", "hexgrid", "freeform", "symmetrical", "old-style revival",
    "clean", "slidershapes", "distance snapped", "iNiS-style", "avant-garde", "perfect stacks",
    "simple", "chaotic", "repetition", "progression", "high contrast",
    "improvisation", "playfield usage", "playfield constraint", "difficulty spike",
    "low sv", "high sv", "colorhax", "tech", "slider tech", "complex sv", "reading",
    "visually dense", "overlap reading", "alt", "jump aim", "sharp aim", "wide aim",
    "linear aim", "aim control", "flow aim", "precision", "finger control",
    "complex snap divisors", "bursts", "streams", "spaced streams", "cutstreams",
    "stamina", "tournament custom"
]

all_descriptors = set(descriptors)
descriptor_history = {}  # key: tuple of descriptors, value: reward

# Q-Table 以 tuple(action) 為 key
q_table = {}

# 訓練參數
episodes = 20
alpha = 0.1
gamma = 0.95
epsilon = 0.2

rewards = []

def select_descriptors(epsilon, descriptor_history, all_descriptors):
    k = random.randint(3, 5)
    if np.random.rand() < epsilon or len(descriptor_history) == 0:
        selected = random.sample(list(all_descriptors), k)
    else:
        best = max(descriptor_history.items(), key=lambda x: x[1])[0]
        selected = list(best)

    neg_candidates = list(all_descriptors - set(selected))
    neg_selected = random.sample(neg_candidates, k)
    return selected, neg_selected

for episode in range(episodes):
    if len(q_table) == 0:
        best_action = [random.choice(vs) for vs in discrete_values]
    else:
        best_action_tuple = max(q_table.items(), key=lambda item: item[1])[0]
        best_action = list(best_action_tuple)

    action = [
        np.random.choice(vs) if np.random.rand() < epsilon else best_action[i]
        for i, vs in enumerate(discrete_values)
    ]
    action_array = np.array(action)
    action_key = tuple(action_array)

    descriptors_selected, negative_descriptors_selected = select_descriptors(
        epsilon, descriptor_history, all_descriptors
    )

    full_action = {
        "parameters": action_array.tolist(),
        "descriptors": descriptors_selected,
        "negative_descriptors": negative_descriptors_selected
    }

    reward = env.step(full_action)

    if action_key not in q_table:
        q_table[action_key] = 0.0

    max_next_q = max(q_table.values(), default=0)
    q_table[action_key] += alpha * (reward + gamma * max_next_q - q_table[action_key])

    descriptor_key = tuple(sorted(descriptors_selected))
    if descriptor_key not in descriptor_history or reward > descriptor_history[descriptor_key]:
        descriptor_history[descriptor_key] = reward

    rewards.append(reward)
    print(f"Full Action: {full_action}")
    print(f"Episode {episode+1}/{episodes}: Reward = {reward:.4f}")

plt.plot(rewards, label='reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.savefig("reward_curve.png")
plt.close()

best_action = max(q_table.items(), key=lambda item: item[1])
print("\n✅ 最佳參數組合：")
print(f"Action: {best_action[0]}")
print(f"Estimated Q-value: {best_action[1]:.4f}")
