# ReinforcementLearning

## 利用强化学习的Q价值迭代,Q学习以及SARSA方法解决小车爬山以及倒立摆的控制问题

1. Python3 运行每个模块

2. 默认是train模式，即先训练再展示结果

3. 也可以修改源码最后一行，根据提示使其只展示训练结果

4. 预先为每个模块提供了预训练的结果，可修改源码最后一行查看预训练结果

5. result文件夹的*sequence 文件是最优策略序列，其中有两行数据，第一行和第二行分别是一个事件从初始状态之后的动作序列, 状态序列,数据之间用空格分开

6. *optimal.npy文件是最优策略的Q矩阵，形状为：[actionSize, state1Size, state2Size]

