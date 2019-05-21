# coding=utf8
# 问题描述：倒立摆是将一个物体固定在一个圆盘的非中心点位置,
#   由直流电机驱动将其在 垂直平面内进行旋转控制的系统 (图 2).
#   由于输入电压是受限的, 因此电机并不能 提供足够的动力直接将摆杆推完一圈.
#   相反, 需要来回摆动收集足够的能量, 然后 才能将摆杆推起并稳定在最高点
# 本函数使用SARSA方法来解决问题
# author: Zhang Chizhan
# date: 2019/5/18

import numpy as np
import matplotlib.pyplot as plt

# 倒立摆角加速度公式
def get_ddalpha(alpha, dalpha, u):
    m = 0.055
    g = 9.81
    l = 0.042
    J = 1.91e-4
    b = 3e-6
    K = 0.0536
    R = 9.5
    return (m*g*l*np.sin(alpha) - b*dalpha-(K**2)*dalpha/R + K*u/R)/J

# 状态转移函数(针对单个数字)  输入:旧角度，旧角速度，动作    输出：新角度，新速度
def get_new_state(old_alpha, old_dalpha, action):
    Ts = 0.005
    new_alpha = old_alpha + Ts * old_dalpha
    new_dalpha = old_dalpha + Ts * get_ddalpha(old_alpha, old_dalpha, action)
    tem = 15 * np.pi
    if new_dalpha < -tem:
        new_dalpha = -tem
    elif new_dalpha > tem:
        new_dalpha = tem
    return new_alpha, new_dalpha

# 状态空间离散化   输入:状态[alpha, dalpha]    输出:对应的状态空间下标
def get_indice(alpha, dalpha, Numa, Numda):
    tem = 15 * np.pi
    normal_a = (alpha + np.pi) % (2 * np.pi) - np.pi
    indice_a = int((normal_a + np.pi)/(2*np.pi) * Numa)
    indice_da = int((dalpha + tem)/(2*tem) * Numda)
    if indice_da == Numda: indice_da -= 1
    return indice_a, indice_da

# 离散空间采样
def get_sample(indice, low, high, Num):
    deta = (high - low) / Num
    sample = (low + deta/2) + deta * indice
    return sample

# epsilon搜索策略  输入：Q函数矩阵,当前的状态空间(p,v), epsilon参数值  输出：epsilon贪心策略动作下标
def get_greedy_act(Q, indice_p, indice_v, epsilon):
    opt_act = np.argmax(Q[:,indice_p,indice_v])
    if np.random.random() > epsilon:
        return opt_act
    else:
        return np.random.choice([0,1,2])

# 奖励函数
def getR(alpha, dalpha, action):
    Rrew = 1
    return -(5*alpha**2+0.1*dalpha**2)-Rrew*action**2

# 存储结果
def saveResult(dic_path, Q, locations, total_acts):
    np.save(dic_path + '/SARSAoptimal.npy', Q)
    sequence_path = dic_path + '/SARSAsequence.txt'
    with open(sequence_path, 'w', encoding='utf8') as f:
        for act in total_acts:
            f.write(str(act) + " ")
        f.write('\n')
        for location in locations:
            f.write(str(location) + ' ')

# 可视化结果
def show_result(path = './result/SARSAsequence.txt'):
    with open(path, 'r', encoding='utf8') as f:
        lines = [line for line in f.readlines()]
    locations = lines[-1].strip().split()
    locations = [float(loc) for loc in locations]
    print("最终距离稳定状态角度误差为:%f rad" % np.abs((locations[-1] + np.pi) % (2*np.pi) - np.pi))
    plt.ion()
    for a in locations:
        plt.cla()
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.title("Inverted Pendulum")
        plt.axis('off')
        x = np.array([0, np.sin(a)])
        y = np.array([0, np.cos(a)])
        plt.plot(x, y)
        plt.plot(x[1],y[1],'ro',markersize = 18)
        plt.pause(0.01)
    plt.ioff()
    plt.show()

def train(Numa, Numda):
    # 以下后缀为a的变量都指摆杆角度，后缀为da的变量都指摆杆的角加速度
    # 状态限制
    lowa, higha = -np.pi, np.pi
    lowda, highda = -15 * np.pi, 15*np.pi

    # 存储所有状态以及所有动作的Q函数矩阵 [action, a, da]
    Q = np.zeros([3, Numa, Numda])

    # 动作矩阵
    actions = np.array([-3, 0, 3])

    # 迭代次数
    N = 20000

    # 初始贪心搜索参数
    epsilon = 1.0

    # 初始学习率
    lr = 1.0

    #epsilon和学习率的衰减率
    decay = 0.9995

    #epsilon的下界
    epsilon_limit = 0.01

    # 折扣因子
    gamma = 0.98

    # 初始状态
    state_a, state_da = -np.pi, 0
    total_step = 0
    optimal = -1e7
    finalerror = 2*np.pi

    # (a, da)距离终止状态(0,0)的误差，用于控制收敛
    controla, controlda = 0.05, 0.01

    # 探索步长限制
    step_control = 300
    # 开始迭代
    for i in range(N):
        a, da, it, total_reward = state_a, state_da, 0, 0
        indice_a, indice_da = get_indice(a, da, Numa, Numda)
        greedy_act = get_greedy_act(Q, indice_a, indice_da, epsilon)
        epsilon = max(decay * epsilon, epsilon_limit)
        lr = decay * lr
        locations, total_acts = [], []
        locations.append(a)
        error_a = np.abs((a-lowa) % (higha - lowa) + lowa)
        while it < step_control and  (error_a > controla or np.abs(da) > controlda):
            it += 1
            total_acts.append(actions[greedy_act])
            newa, newda = get_new_state(a, da, actions[greedy_act])
            indice_newa, indice_newda = get_indice(newa, newda, Numa, Numda)
            greedy_newact = get_greedy_act(Q, indice_newa, indice_newda, epsilon)
            reward = getR(a, da, actions[greedy_act])
            deta = reward + gamma * Q[greedy_newact][indice_newa][indice_newda] - Q[greedy_act][indice_a][indice_da]
            Q[greedy_act][indice_a][indice_da] += lr * deta
            a, da, indice_a, indice_da, greedy_act = newa, newda, indice_newa, indice_newda, greedy_newact
            locations.append(a)
            total_reward += reward
            error_a = np.abs((a-lowa) % (higha - lowa) + lowa)
        total_step += it
        # 若该次迭代到达终点并且误差更小，或者奖励更高，则存储
        if it < step_control:
            if error_a < finalerror or(error_a == finalerror and total_reward > optimal):
                optimal, finalerror = total_reward, error_a
            saveResult('./result', Q, locations, total_acts)
        # 打印迭代信息
        if (i+1) % 100 == 0:
            print("第%d-th | %d 迭代平均步长为 %.2f | %d" % (i+1, N, total_step/100, step_control))
            total_step = 0
    show_result()


if __name__ == '__main__':
    # 这是最优训练结果
    # show_result('./result/pre_trained_SARSAsequence.txt')

    # Numa是摆杆的角度空间划分数量  Numda是摆杆角速度空间划分数量 可以修改该参数
    train(Numa = 200, Numda = 200)

    # 仅显示训练结果
    # show_result()
