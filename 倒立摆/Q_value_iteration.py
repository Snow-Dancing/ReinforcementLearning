# coding=utf8
# 问题描述：倒立摆是将一个物体固定在一个圆盘的非中心点位置,
#   由直流电机驱动将其在 垂直平面内进行旋转控制的系统 (图 2).
#   由于输入电压是受限的, 因此电机并不能 提供足够的动力直接将摆杆推完一圈.
#   相反, 需要来回摆动收集足够的能量, 然后 才能将摆杆推起并稳定在最高点
# 本函数使用Q价值迭代来解决问题
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
# 状态转移函数(针对Numpy矩阵)  输入:旧角度[Numa,Numda],旧角速度[Numa, Numda]，动作    输出：新角度，新速度
def get_new_state_for_matrix(old_alpha, old_dalpha, action):
    Ts = 0.005
    new_alpha = old_alpha + Ts * old_dalpha
    new_dalpha = old_dalpha + Ts * get_ddalpha(old_alpha, old_dalpha, action)
    tem = 15 * np.pi
    new_dalpha[new_dalpha < -tem] = -tem
    new_dalpha[new_dalpha > tem] = tem
    return new_alpha, new_dalpha

# 状态空间离散化   输入:状态[alpha, dalpha]    输出:对应的状态空间下标
def get_indice(alpha, dalpha, Numa, Numda):
    tem = 15 * np.pi
    normal_a = (alpha + np.pi) % (2 * np.pi) - np.pi
    indice_a = int((normal_a + np.pi)/(2*np.pi) * Numa)
    indice_da = int((dalpha + tem)/(2*tem) * Numda)
    if indice_da == Numda: indice_da -= 1
    return indice_a, indice_da

def get_indice_for_matrix(alpha, dalpha, Numa, Numda):
    tem = 15 * np.pi
    normal_a = (alpha + np.pi) % (2 * np.pi) - np.pi
    indice_a = ((normal_a + np.pi)/(2*np.pi) * Numa).astype(int)
    indice_da = ((dalpha + tem)/(2*tem) * Numda).astype(int)
    indice_da[indice_da == Numda] -= 1
    return indice_a, indice_da

# 离散空间采样
def get_sample(indice, low, high, Num):
    deta = (high - low) / Num
    sample = (low + deta/2) + deta * indice
    return sample

# 奖励函数
def getR(alpha, dalpha, action):
    Rrew = 1
    return -(5*alpha**2+0.1*dalpha**2)-Rrew*action**2

# 存储结果
def save_result(total_acts, locations, Numa, Numda):
    path = "./result/Q_value_it_sequence" + str(Numa) + "_" + str(Numda) + ".txt"
    with open(path, 'w', encoding='utf8') as f:
        for act in total_acts:
            f.write(str(act) + " ")
        f.write('\n')
        for loc in locations:
            f.write(str(loc) + " ")
# 可视化结果
def show_result(Numa=2000, Numda=2000):
    path = "./result/Q_value_it_sequence" + str(Numa) + "_" + str(Numda) + ".txt"
    with open(path, 'r', encoding='utf8') as f:
        lines = [line for line in f.readlines()]
    locations = lines[-1].strip().split()
    locations = [float(loc) for loc in locations]
    locations = locations[:300] # 由于最后摆杆在顶端略微摆动，所以只取300个点
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

    # 存储每个空间的a, da指标的矩阵  该矩阵主要是为了利用numpy矩阵运算加速运行
    Matrixa = np.zeros([Numa, Numda])   # 存储每个状态的a指标
    Matrixda = np.zeros([Numa, Numda])  # 存储每个状态的da指标
    for i in range(Numa):
        for j in range(Numda):
            Matrixa[i][j] = i
            Matrixda[i][j] = j

    # 存储所有状态以及所有动作的Q函数矩阵 [action, a, da]
    Q = np.zeros([3, Numa, Numda])

    # 存储前一个状态的Q函数矩阵，用于控制收敛
    Qpre = np.zeros(Q.shape)
    Qpre[0][0][0] = -100  # 满足第一次迭代

    # 动作矩阵
    action = np.array([-3, 0, 3])

    # 迭代次数
    it = 0

    # 折扣因子
    gamma = 0.98

    # 开始迭代
    control = 1e-3  # 控制收敛
    while np.max(np.abs(Q-Qpre)) >= control:
        it += 1
        Qpre = Q.copy()
        maxQpre = np.max(Qpre, axis=0) # 每个状态对应的最大Q(s,a)值,shape:[Numa, Numda]
        sama= get_sample(Matrixa, lowa, higha, Numa) # 每个状态采样摆杆角度
        samda= get_sample(Matrixda, lowda, highda, Numda)   # 每个状态采样摆杆角加速度
        for act in range(action.size):
            new_a, new_da = get_new_state_for_matrix(sama, samda, action[act])
            indice_a, indice_da = get_indice_for_matrix(new_a, new_da, Numa, Numda)
            Q[act] = getR(sama, samda, action[act]) + gamma * maxQpre[indice_a,indice_da]
        if it % 10 == 0:
            print("已经进行了%d次迭代, 当前误差为%f | %f" % (it, np.max(np.abs(Q-Qpre)), control))
    print("迭代结束！一共进行了%d次迭代" % (it-1))

    # 根据迭代之后的Q价值矩阵求出最终的最优策略
    a = state_a = -np.pi  # 从初始状态出发
    da = state_da = 0
    maxQ = np.argmax(Q, axis=0) # 每个状态的最优动作, shape:[Nump+1, Numv]
    total_act = []  # 存储从初始状态之后每一步的动作
    location = []   # 存储从初状态之后每一步的摆杆角度值
    location.append(a)
    indice_a, indice_da = get_indice(a, da, Numa, Numda) # 状态对应的空间下标
    it = 0
    len_limit_a = higha - lowa
    control = 0.01  # 控制收敛  但此值偏小而从未达到过，建议使用0.05
    while np.abs((a-lowa) % len_limit_a + lowa) > control or np.abs(da) > control:
        it += 1
        act = maxQ[indice_a][indice_da]  # 选取最优动作
        total_act.append(action[act])   # 存储动作
        a, da = get_new_state(a, da, action[act]) # 得到新的状态
        location.append(a)
        indice_a, indice_da = get_indice(a, da, Numa, Numda)
        # 最多从初始状态寻找1000步,若此时仍未收敛，则退出循环
        if it > 1000:
            break
    np.save("./result/Q_value_it_optimal.npy", Q)
    save_result(total_act, location, Numa, Numda)
    show_result(Numa, Numda)

if __name__ == '__main__':
    # 这是最优训练结果
    # show_result(Numa =2000, Numda = 2000)
    # 给出以下预先训练好的不同划分空间数量的结果,可以任意选择一行运行，运行前请注释掉前一个运行的代码
    # show_result(Numa = 500, Numda = 500)
    # show_result(Numa = 600, Numda = 600)
    # show_result(Numa = 700, Numda = 700)
    # show_result(Numa = 800, Numda = 800)
    # show_result(Numa = 900, Numda = 900)
    # show_result(Numa = 1000, Numda = 1000)
    # show_result(Numa = 2000, Numda = 2000)
    # show_result(Numa = 3000, Numda = 3000)

    # Numa是摆杆的角度空间划分数量  Numda是摆杆角速度空间划分数量 可以修改该参数
    #  注:训练可能有些慢
    train(Numa = 500, Numda = 500)
