# coding=utf-8
# 问题描述： 小车爬山
# -1.2 <= p <= 0.5; -0.07 <= v <= 0.07
# 高度 = sin(3p)
# 状态转换：
#   v[t+1] = bound(v[t] + 0.001a[t] + g*cos(3p[t]))
#   p[t+1] = bound(p[t] + v[t+1])
# 其中g = -0.0025对应重力因素, bound函数将输入变量截断在允许范围内
# 如果p[t+1]被截断了，v[t+1]也相应地被置为0
# 奖励定为每一步为-1
# 当p[t+1] > 0.5时，小车已经达到了目标点，即达到了终止状态
# 本函数提供一种Q学习的方法求解最优策略  状态空间为20×20
# author: Zhang Chizhan
# date: 2019/5/20

import numpy as np
import matplotlib.pyplot as plt

# 状态转移函数  输入：oldp, oldv     输出：newp, newv
def get_new_state(oldp, oldv, action):
    g = -0.0025
    newv = oldv + 0.001 * action + g * np.cos(3 * oldp)
    if newv < -0.07:
        newv = -0.07
    elif newv > 0.07:
        newv = 0.07
    newp = oldp + newv
    if newp < -1.2 and newv < 0:
        newp = -1.2
        newv = 0
    return newp, newv

# 状态空间离散化   输入：p,v      输出:p对应状态空间的下标， v对应状态空间的下标
def get_indice(p ,v, Nump, Numv):
    indice_p = int((p + 1.2)/1.7 * Nump)
    indice_v = int((v + 0.07)/0.14 * Numv)
    if indice_v == Numv:
        indice_v -= 1
    if indice_p == Nump:
        indice_p -= 1
    return indice_p, indice_v

# epsilon搜索策略  输入：Q函数矩阵,当前的状态空间(p,v), epsilon参数值  输出：epsilon贪心策略动作下标
def get_greedy_act(Q, indice_p, indice_v, epsilon):
    opt_act = np.argmax(Q[:,indice_p,indice_v])
    if np.random.random() > epsilon:
        return opt_act
    else:
        return np.random.choice([0,1,2])

# 根据最终的Q矩阵得到状态及动作序列
def get_sequence(Q=[], actions=[1,0,-1], Nump=20, Numv=20, reward=-1, gamma=1):
    p, v = -0.5, 0
    epsilon = lr = 0.001
    indice_p, indice_v = get_indice(p, v, Nump, Numv)
    locations, total_acts = [], []
    locations.append(p)
    while p < 0.5:
        greedy_act = get_greedy_act(Q, indice_p, indice_v, epsilon)
        total_acts.append(actions[greedy_act])
        newp, newv = get_new_state(p, v, actions[greedy_act])
        indice_newp, indice_newv = get_indice(newp, newv, Nump, Numv)
        max_newQa = np.max(Q[:, indice_newp, indice_newv])
        deta = reward + gamma * max_newQa - Q[greedy_act][indice_p][indice_v]
        Q[greedy_act][indice_p][indice_v] = Q[greedy_act][indice_p][indice_v] + lr * deta
        p, v, indice_p, indice_v = newp, newv, indice_newp, indice_newv
        locations.append(p)
    return locations, total_acts

# 存储结果  存储最优Q矩阵和最优坐标序列
def saveResult(dic_path, Q, locations, total_acts):
    np.save(dic_path + '/QLoptimal.npy', Q)
    sequence_path = dic_path + '/QLsequence.txt'
    with open(sequence_path, 'w', encoding='utf8') as f:
        for act in total_acts:
            f.write(str(act) + " ")
        f.write('\n')
        for location in locations:
            f.write(str(location) + ' ')

# 结果可视化 输入:p的序列
def show_result(path = './result/QLsequence.txt'):
    with open(path, 'r', encoding='utf8') as f:
        lines = [line for line in f.readlines()]
    locations = lines[-1].strip().split()
    locations = [float(loc) for loc in locations]
    print("总代价为:%d" % -(len(locations) - 1))
    x = np.arange(-1.3, 0.6, 0.01)
    y = np.sin(3 * x)
    p = np.array(locations)
    hp = np.sin(3 * p)
    plt.ion()
    for i in range(len(p)):
        plt.cla()
        plt.title("MountCar")
        plt.axis('off')
        plt.plot(x, y)
        plt.plot(p[i],hp[i],'ro',markersize = 18)
        plt.pause(0.01)
    plt.ioff()
    plt.show()

def train():
    # 状态空间离散化的大小
    Nump = 20
    Numv = 20

    # 存储所有状态以及所有动作的Q函数矩阵  Q[state, action] : [action, p, v]
    Q = np.zeros([3, Nump, Numv])

    # 动作矩阵
    actions = np.array([1, 0, -1])

    # 迭代次数
    N = 20000

    # 初始贪心搜索参数
    epsilon = 1.0

    #epsilon的下界
    epsilon_limit = 0.01

    # 初始学习率
    lr = 1.0

    #epsilon和学习率的衰减率
    decay = 0.9995

    # 折扣因子
    gamma = 1

    # 奖励函数
    reward = -1

    # 初始状态
    state_p, state_v = -0.5, 0
    total_step = 0
    optimal = -1e5   # 存储当前最优价值

    # 探索步长限制
    step_control = 200
    # 开始迭代
    for i in range(N):
        # 每次迭代状态初始化
        p, v, it, total_reward = state_p, state_v, 0, 0
        indice_p, indice_v = get_indice(p, v, Nump, Numv)   # 得到状态空间下标
        # 参数衰减
        epsilon = max(decay * epsilon, epsilon_limit)
        lr = decay * lr
        # 记录角度序列和动作序列的列表
        locations, total_acts = [], []
        locations.append(p)
        while it < step_control:
            it += 1
            greedy_act = get_greedy_act(Q, indice_p, indice_v, epsilon)
            total_acts.append(actions[greedy_act])
            newp, newv = get_new_state(p, v, actions[greedy_act])
            if newp > 0.5:  # 达到终止状态
                locations.append(newp)
                break
            indice_newp, indice_newv = get_indice(newp, newv, Nump, Numv)
            max_newQa = np.max(Q[:, indice_newp, indice_newv])
            deta = reward + gamma * max_newQa - Q[greedy_act][indice_p][indice_v]
            Q[greedy_act][indice_p][indice_v] = Q[greedy_act][indice_p][indice_v] + lr * deta
            p, v, indice_p, indice_v = newp, newv, indice_newp, indice_newv
            locations.append(p)
            total_reward += reward
        total_step -= it
        if total_reward > optimal:  # 若当前价值更高，则存储
            optimal = total_reward
            saveResult('./result', Q, locations, total_acts)
        if (i+1) % 100 == 0:
            print("第%d-th | %d 迭代平均代价为%.2f" % (i+1, N, total_step/100))
            total_step = 0
    show_result()

if __name__ == '__main__':
    train()

    # 展示预先训练好的结果
    # show_result('./result/pre_trained_QLsequence.txt')

    # 展示自己训练之后的结果
    # show_result()

