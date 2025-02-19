import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import random


# 超参数配置
class Config:
    N_RICES = 8  # 稻子数量
    TRAIN_EPISODES = 5000  # 训练轮次
    TEST_EPISODES = 500  # 测试次数
    BATCH_SIZE = 128  # 经验回放批次大小
    MEMORY_SIZE = 10000  # 经验池容量
    GAMMA = 0.97  # 折扣因子
    LR = 0.001  # 学习率
    INIT_EPSILON = 0.6  # 初始探索率
    MIN_EPSILON = 0.02  # 最小探索率
    EPS_DECAY = 0.9995  # 探索率衰减
    TIME_PENALTY = -0.02  # 时间惩罚系数


# 深度Q网络架构
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# 强化学习环境
class SecretaryEnv:
    def __init__(self):
        self.rices = []
        self.current_step = 0
        self.observed_max = 0
        self.global_max = 0

    def reset(self, custom_sequence=None):
        """初始化或重置环境"""
        if custom_sequence:
            self.rices = custom_sequence.copy()
        else:
            # 生成随机稻子序列（训练时自动打乱）
            self.rices = [round(random.uniform(1, 10), 2) for _ in range(Config.N_RICES)]
            random.shuffle(self.rices)

        self.global_max = max(self.rices)
        self.current_step = 0
        self.observed_max = 0
        return self._get_state()

    def _get_state(self):
        """状态特征工程"""
        current_value = self.rices[self.current_step]
        is_best = 1.0 if current_value > self.observed_max else 0.0
        progress = self.current_step / (Config.N_RICES - 1)
        return [is_best, progress]

    def step(self, action):
        """执行动作并返回新状态、奖励、终止标志"""
        current_value = self.rices[self.current_step]
        self.observed_max = max(self.observed_max, current_value)

        if action == 1:  # 选择当前稻子
            reward = 5.0 if current_value == self.global_max else -2.0
            return None, reward, True

        self.current_step += 1
        if self.current_step >= Config.N_RICES - 1:  # 强制最终选择
            final_value = self.rices[-1]
            reward = 5.0 if final_value == self.global_max else -2.0
            return None, reward, True

        return self._get_state(), Config.TIME_PENALTY, False


# 智能体实现
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LR)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.loss_fn = nn.SmoothL1Loss()
        self.epsilon = Config.INIT_EPSILON

    def select_action(self, state):
        """ε-贪婪策略"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            return self.policy_net(state_tensor).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到记忆池"""
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self):
        """执行经验回放更新"""
        if len(self.memory) < Config.BATCH_SIZE:
            return

        # 从记忆池中采样
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # 处理next_states中的None值
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([s for s in next_states if s is not None])

        # 计算Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        next_q = torch.zeros(Config.BATCH_SIZE)
        with torch.no_grad():
            next_q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        target_q = rewards + Config.GAMMA * next_q * (~dones)

        # 计算损失
        loss = self.loss_fn(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(Config.MIN_EPSILON, self.epsilon * Config.EPS_DECAY)

    def sync_target_net(self):
        """同步目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练过程
def train_agent():
    env = SecretaryEnv()
    agent = DQNAgent()

    print("开始训练...")
    for episode in range(1, Config.TRAIN_EPISODES + 1):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # 执行模型更新
            agent.update_model()

            if done:
                break

        # 定期同步目标网络
        if episode % 10 == 0:
            agent.sync_target_net()

        # 更新探索率
        agent.update_epsilon()

        # 输出训练进度
        if episode % 500 == 0:
            print(f"Episode {episode} | 平均奖励: {total_reward:.2f} | ε: {agent.epsilon:.3f}")

    print("训练完成！")
    return agent


# 自动化测试与分析
def comprehensive_test(agent):
    env = SecretaryEnv()
    selection_counts = defaultdict(int)
    correct_counts = 0

    print("\n进行严格测试...")
    for _ in range(Config.TEST_EPISODES):
        # 生成测试序列（保持原始顺序）
        sequence = [round(random.uniform(1, 10), 2) for _ in range(Config.N_RICES)]
        true_max = max(sequence)

        state = env.reset(custom_sequence=sequence)
        chosen_index = -1

        while True:
            with torch.no_grad():
                action = agent.policy_net(torch.FloatTensor(state)).argmax().item()

            if action == 1:
                chosen_index = env.current_step
                break

            next_state, _, done = env.step(action)
            if done:
                chosen_index = Config.N_RICES - 1
                break
            state = next_state

        # 记录结果
        selection_counts[chosen_index] += 1
        if sequence[chosen_index] == true_max:
            correct_counts += 1

    # 输出分析报告
    success_rate = correct_counts / Config.TEST_EPISODES
    print(f"\n测试结果（{Config.TEST_EPISODES}次实验）")
    print(f"成功选中最长稻子的概率: {success_rate:.2%}")
    print("选择位置分布:")

    total = sum(selection_counts.values())
    for pos in range(Config.N_RICES):
        count = selection_counts.get(pos, 0)
        print(f"位置 {pos + 1}: {count:3d}次 ({count / total:.1%})")

    # 理论参考值
    k = int(Config.N_RICES / np.e)
    theoretical_rate = sum(1 / (k + i) for i in range(Config.N_RICES - k)) / Config.N_RICES
    print(f"\n理论最优成功率: {theoretical_rate:.2%}")


if __name__ == "__main__":
    trained_agent = train_agent()
    comprehensive_test(trained_agent)