#coding=utf-8
# 问题描述： 小车爬山
#         -1.2 <= p <= 0.5; -0.07 <= v <= 0.07
#         高度 = sin(3p)
#         状态转换：
#           v[t+1] = bound(v[t] + 0.001a[t] + g*cos(3p[t]))
#           p[t+1] = bound(p[t] + v[t+1])
#         其中g = -0.0025对应重力因素, bound函数将输入变量截断在允许范围内
#         如果p[t+1]被截断了，v[t+1]也相应地被置为0
#         奖励定为每一步为-1
#         当p[t+1] > 0.5时，小车已经达到了目标点，即达到了终止状态
# 本函数描述：
#         提供一种Q函数价值迭代的方法求解最优策略
#          由于是Q价值迭代，所以状态空间划分较细，为200×200,
#         由于使用了Numpy矩阵乘法加速运算，所以该方法收敛速度较快
# author: Zhang Chizhan
# date: 2019/5/17

import numpy as np
from matplotlib import pyplot as plt

# 状态空间采样   每个状态空间取中间值
def get_sample(indice, low, high, Num):
    deta = (high - low) / Num
    sample = (low + deta/2) + deta * indice
    return sample

# 状态转移函数  输入：oldp, oldv     输出：newp, newv
def get_new_state(oldp, oldv, action):
    g = -0.0025
    newv = oldv + 0.001 * action + g * np.cos(3 * oldp)
    if newv < -0.07:
        newv = -0.07
    elif newv > 0.07:
        newv = 0.07
    newp = oldp + newv
    if newp < -1.2:
        newp = -1.2
        newv = 0

    return newp, newv

# 状态空间离散化   输入：p,v    输出:p对应状态空间的下标， v对应状态空间的下标
def get_indice(p ,v, Nump, Numv):
    indice_p = int((p + 1.2)/1.7 * Nump)
    indice_v = int((v + 0.07)/0.14 * Numv)
    if indice_v == Numv:
        indice_v -= 1
    if p == 0.5:
        indice_p = Nump-1
    elif p > 0.5:
        indice_p = Nump
    return indice_p, indice_v

# 状态转移函数,该函数专供numpy矩阵乘法加速运算所用  oldp,oldv为矩阵
def get_new_state_for_matrix(oldp, oldv, action):
    g = -0.0025
    newv = oldv + 0.001 * action + g * np.cos(3 * oldp)
    newv[newv < -0.07] = -0.07
    newv[newv > 0.07] = 0.07
    newp = oldp + newv
    newv[newp < -1.2] = 0
    newp[newp < -1.2] = -1.2
    return newp, newv

# 状态空间离散化,该函数专供numpy矩阵乘法加速运算所用 p, v为矩阵
def get_indice_for_matrix(p ,v, Nump, Numv):
    indice_p = ((p + 1.2)/1.7 * Nump).astype(int)
    indice_v = ((v + 0.07)/0.14 * Numv).astype(int)
    indice_v[indice_v == Numv] -= 1
    indice_p[p==0.5] = Nump-1
    indice_p[p>0.5] = Nump
    return indice_p, indice_v

# 存储结果  存储最优Q矩阵和最优坐标序列
def saveResult(dic_path, Q, locations, total_acts):
    np.save(dic_path + '/Qvalue_it_optimal.npy', Q)
    sequence_path = dic_path + '/Qvalue_it_sequence.txt'
    with open(sequence_path, 'w', encoding='utf8') as f:
        for act in total_acts:
            f.write(str(act) + " ")
        f.write('\n')
        for location in locations:
            f.write(str(location) + ' ')

# 结果可视化 输入:p的序列
def show_result(path = './result/Qvalue_it_sequence.txt'):
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
    Nump = 200
    Numv = 200

    # 存储所有状态以及所有动作的Q函数矩阵  Q[state, action] : [action, p, v]
    Q = np.zeros([3, Nump + 1, Numv])   # 此处p多一维是为了记录终止状态

    # 存储前一个状态的Q函数矩阵，用于控制收敛
    Qpre = np.zeros(Q.shape)
    Qpre[0][0][0] = -100  # 满足第一次迭代

    # 状态限制
    lowp, highp = -1.2, 0.5
    lowv, highv = -0.07, 0.07
    # 存储每个空间的p, v指标的矩阵
    Matrixp = np.zeros([Nump, Numv])
    Matrixv = np.zeros([Nump, Numv])
    for i in range(Nump):
        for j in range(Numv):
            Matrixp[i][j] = i
            Matrixv[i][j] = j

    # 动作矩阵
    action = np.array([1, 0, -1])

    # 迭代次数
    it = 0

    # 折扣因子
    gamma = 1

    # 控制收敛参数
    control = 1e-3
    # 开始迭代
    while np.max(np.abs(Q-Qpre)) >= control:
        it += 1
        Qpre = Q.copy()
        maxQpre = np.max(Qpre, axis=0) # 每个状态对应的最大Q(s,a)值,shape:[Nump+1, Numv]
        samp= get_sample(Matrixp, lowp, highp, Nump)
        samv= get_sample(Matrixv, lowv, highv, Numv)
        for act in range(action.size):
            new_p, new_v = get_new_state_for_matrix(samp, samv, action[act])
            indice_p, indice_v = get_indice_for_matrix(new_p, new_v, Nump, Numv)
            Q[act,:Nump,:] = -1 + gamma * maxQpre[indice_p,indice_v]
        if it % 10 == 0:
            print("已经进行了%d次迭代, 当前误差为%.2f | %f" % (it,np.max(np.abs(Q-Qpre)), control))
    print("迭代结束！一共进行了%d次迭代" % (it-1))
    # 根据迭代之后的Q价值矩阵求出最终的最优策略
    p, v = -0.5, 0  # 从初始状态出发
    maxQ = np.argmax(Q, axis=0) # 每个状态的最优动作, shape:[Nump+1, Numv]
    total_acts, locations = [], []
    locations.append(p) # 存储状态p值
    indice_p, indice_v = get_indice(p, v, Nump, Numv) # 状态对应的空间下标
    while p <= 0.5:
        act = maxQ[indice_p][indice_v]  # 选取最优动作
        total_acts.append(action[act])   # 存储动作
        p, v = get_new_state(p, v, action[act]) # 得到新的状态
        locations.append(p)
        indice_p, indice_v = get_indice(p, v, Nump, Numv)
    saveResult('./result', Q, locations, total_acts)
    show_result()

if __name__ == '__main__':
    # 训练
    train()

    # 仅仅展示自己训练之后的结果
    # show_result()

    # 下边是预训练好的数据
    # show_result('./result/pre_trained_Qvalue_it_sequence.txt')
