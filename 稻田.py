import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 生成训练数据
def generate_training_data(num_samples=1000, num_rices=8, length_range=(1, 10), step=0.01):
    data = []
    for _ in range(num_samples):
        lengths = np.random.uniform(length_range[0], length_range[1], num_rices)
        lengths = np.round(lengths / step) * step
        data.append(lengths)
    return data

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# 定义环境和奖励机制
class RiceField:
    def __init__(self, lengths):
        self.lengths = lengths
        self.index = 0

    def step(self):
        if self.index < len(self.lengths):
            length = self.lengths[self.index]
            self.index += 1
            return length
        else:
            return None

    def reset(self):
        self.index = 0

def calculate_reward(chosen_length, max_length):
    return 1 - abs(chosen_length - max_length) / max_length

# 训练策略网络
def train_policy_network(policy_net, training_data, epochs=1000, learning_rate=0.01):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for lengths in training_data:
            env = RiceField(lengths)
            log_probs = []
            chosen_length = None

            while True:
                length = env.step()
                if length is None:
                    break

                state = torch.tensor([length], dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state)
                action = torch.multinomial(probs, 1).item()

                if action == 1 and chosen_length is None:
                    chosen_length = length

                log_prob = torch.log(probs[0, action])
                log_probs.append(log_prob)

            if chosen_length is None:
                chosen_length = lengths[-1]

            reward = calculate_reward(chosen_length, max(lengths))
            cumulative_rewards = torch.tensor([reward] * len(log_probs), dtype=torch.float32)

            policy_loss = []
            for log_prob, R in zip(log_probs, cumulative_rewards):
                policy_loss.append(-log_prob * R)

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {policy_loss.item()}")

# 测试策略网络
def test_policy_network(policy_net):
    lengths = []
    print("请输入8根稻子的长度：")
    for i in range(8):
        length = float(input(f"稻子 {i+1} 的长度: "))
        lengths.append(length)

    env = RiceField(lengths)
    chosen_length = None

    while True:
        length = env.step()
        if length is None:
            break

        state = torch.tensor([length], dtype=torch.float32).unsqueeze(0)
        probs = policy_net(state)
        action = torch.multinomial(probs, 1).item()

        if action == 1 and chosen_length is None:
            chosen_length = length

    if chosen_length is None:
        chosen_length = lengths[-1]

    print(f"计算机选择的稻子长度为: {chosen_length}")
    print(f"该组稻子的最大长度为: {max(lengths)}")
    print(f"选择的稻子长度与最大长度的差值为: {abs(chosen_length - max(lengths))}")

# 主程序
if __name__ == "__main__":
    training_data = generate_training_data()
    policy_net = PolicyNetwork(input_dim=1, hidden_dim=16, output_dim=2)
    train_policy_network(policy_net, training_data)
    test_policy_network(policy_net)