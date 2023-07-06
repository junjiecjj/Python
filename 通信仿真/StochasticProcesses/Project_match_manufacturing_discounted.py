import numpy as np
import pandas as pd

class batch_discount_MDP:
    def __init__(self, statuses_set = None, action_set = None, num_states = None, param_list = None):
        # initialize statuses set
        if statuses_set is None:
            self.statuses_set = [n for n in range(num_states)]
        else:
            self.statuses_set = statuses_set

        self.status_init = None
        self.status_terminal = None

        # initialize action set
        if action_set is None:
            self.actions_set = [0, 1]
        else:
            self.actions_set = action_set

        self.action = None
        self.param_list = param_list
        self.policy = {}
        self.trans_mats = None
        self.J_series = []
        self.J_optimal = []
        self.action_series = []

        if num_states is None:
            self.num_states = len(self.statuses_set)
        else:
            self.num_states = num_states

    def generate_trans_graph(self):
        start_set = self.statuses_set
        end_set = self.statuses_set

        # 不同动作对应不同概率转移矩阵
        trans_mats = {}
        for action_ in self.actions_set:
            trans_mats[action_] = np.zeros((len(start_set), len(end_set)))
            if action_ == 0:
                for i in start_set:
                    if i < self.param_list["n"]:
                        trans_mats[action_][i, i + 1] = self.param_list["p"]
                        trans_mats[action_][i, i] = 1 - self.param_list["p"]

                # 转化成DataFrame
                trans_mats[action_] = pd.DataFrame(trans_mats[action_])
                trans_mats[action_].index = start_set
                trans_mats[action_].columns = end_set

            elif action_ == 1:
                for i in start_set:
                    trans_mats[action_][i, 1] = self.param_list["p"]
                    trans_mats[action_][i, 0] = 1 - self.param_list["p"]

                # trans_mats[action_][0, 1] = 0
                # trans_mats[action_][0, 0] = 0
                trans_mats[action_][self.param_list["n"], 1] = self.param_list["p"]
                trans_mats[action_][self.param_list["n"], 0] = 1 - self.param_list["p"]

                # 转化成DataFrame
                trans_mats[action_] = pd.DataFrame(trans_mats[action_])
                trans_mats[action_].index = start_set
                trans_mats[action_].columns = end_set

        # create_dense_edge(start_set, end_set, Matrix=trans_mats[0])
        # create_dense_edge(start_set, end_set, Matrix=trans_mats[1])
        self.trans_mats = trans_mats

    def per_stage_cost(self, status, action):
        if action != 1:
            return status * self.param_list["c"]
        elif action == 1:
            return self.param_list["K"]

    def value_iteration(self, num_states = None, init_J = None):
        """
        batch_discount_MDP的值迭代解法，
        :param num_states: 最大阶段数 （即最大迭代次数）
        :param init_J: J的初始态 (一维向量，长度和状态量相同)
        :return: 最优J，最优策略，分别记录在类中。
        """
        if num_states is None:
            num_states = self.num_states

        # 生成全0的J序列 (K-by-num_statuses)：
        J_series = np.zeros((num_states + 1, len(self.statuses_set)))
        if init_J is not None:
            J_series[0, :] = init_J

        # 生成全0的动作序列 (1-by-num_statuses):
        action_series = np.zeros((num_states, len(self.statuses_set)))
        # 开始阶段迭代
        for k in range(num_states):
            # 在每个阶段，对每个状态分别计算当前的optimal cost
            for status_ in self.statuses_set:
                # 如果订单量在可积压范围内：
                if status_ != self.param_list["n"]:
                    # 计算所有可能的Pattern
                    Pt = {}
                    for action_ in self.actions_set:
                        if action_ == 1:
                            Pt[action_] = self.per_stage_cost(status=status_,action=action_) + self.param_list["alpha"] * self.trans_mats[action_].iloc[status_, :] @ J_series[k, :].T
                        elif action_ == 0:
                            Pt[action_] = self.per_stage_cost(status=status_,action=action_) + self.param_list["alpha"] * self.trans_mats[action_].iloc[status_, :] @ J_series[k, :].T
                    # 产生下一阶段的optimal cost，即对应的pattern
                    J_series[k+1, status_] = min(Pt.values())
                    # 记录当前阶段的动作
                    action_series[k, status_] = min(Pt, key = Pt.get)

                # 如果订单量不再能够积压：
                if status_ == self.param_list["n"]:
                    # 则当前动作必须为处理！
                    action_ = 1
                    # 产生下一阶段的optimal cost，即对应的pattern
                    J_series[k+1, status_] = self.per_stage_cost(status=status_,action=action_) + self.param_list["alpha"] * self.trans_mats[action_].iloc[status_, :] @ J_series[k, :].T
                    # 记录当前阶段的动作
                    action_series[k, status_] = action_

            # 最优性检验：
            residual = np.linalg.norm(J_series[k+1, :] - J_series[k, :])
            if residual < 1e-7:
                print("J has converged!")
                break
        self.J_optimal = J_series[k+1, :]
        self.policy = action_series[k, :]
        self.J_series = J_series
        self.action_series = action_series

    def policy_iteration(self, num_states = None, init_policy = None):
        if num_states is None:
            num_states = self.num_states
        if init_policy is None:
            # init_policy = np.random.randint(0, 2, len(self.statuses_set))
            init_policy = np.ones(len(self.statuses_set))
            init_policy[0:1] = 0

        # 生成全0的J序列 (K-by-num_statuses)：
        J_series = np.zeros((num_states + 1, len(self.statuses_set)))

        # 生成全0的动作序列 (1-by-num_statuses):
        action_series = np.zeros((num_states, len(self.statuses_set)))
        action_series[0, :] = init_policy

        # 开始迭代
        for k in range(num_states):
            # 构造Linear System
            b = np.zeros(len(self.statuses_set))
            A = np.zeros((len(self.statuses_set),len(self.statuses_set)))
            i = 0
            for status_ in self.statuses_set:
                action_ = action_series[k, i]
                b[i] = self.per_stage_cost(status=status_, action=action_)
                A[i, :] = self.param_list["alpha"] * self.trans_mats[action_].iloc[[i], :]
                i += 1

            # 构造完毕，解方程：
            J_series[k, :] = np.linalg.solve(a=A - np.eye(len(self.statuses_set)), b= - b)

            # 更新策略
            for status_ in self.statuses_set:
                if status_ != self.param_list["n"]:
                    # 计算所有可能的Pattern
                    Pt = {}
                    for action_ in self.actions_set:
                        if action_ == 1:
                            Pt[action_] = self.per_stage_cost(status=status_, action=action_) + self.param_list["alpha"] * \
                                          self.trans_mats[action_].iloc[status_, :] @ J_series[k, :].T
                        elif action_ == 0:
                            Pt[action_] = self.per_stage_cost(status=status_, action=action_) + self.param_list["alpha"] * \
                                          self.trans_mats[action_].iloc[status_, :] @ J_series[k, :].T
                    # 产生下一阶段的动作
                    action_series[k+1, status_] = min(Pt, key=Pt.get)

                # 如果订单量不再能够积压：
                if status_ == self.param_list["n"]:
                    # 则当前动作必须为处理！
                    action_ = 1

                    # 产生下一阶段的动作
                    action_series[k+1, status_] = action_

            # 最优性检验：
            residual = 1.
            if k>=1:
                residual = np.linalg.norm(J_series[k, :] - J_series[k-1, :])
            if residual < 1e-7:
                print("J has converged!")
                break
        self.J_optimal = J_series[k + 1, :]
        self.policy = action_series[k, :]
        self.J_series = J_series
        self.action_series = action_series


# parameters:
c = 1
K = 5
n = 10
p = 0.5
alpha = 0.9
param_list = {}
param_list["c"] = c
param_list["K"] = K
param_list["n"] = n
param_list["p"] = p
param_list["alpha"] = alpha

# action space
action_set = [0, 1]
# status space
statuses_set = [n for n in range(n+1)]

# generate the corresponding markov process
batch_discount = batch_discount_MDP(statuses_set=statuses_set, action_set=action_set, param_list=param_list)
# generate probability transport graph
batch_discount.generate_trans_graph()
# generate policy
for status_ in statuses_set:
    if status_ != param_list["n"]:
        batch_discount.policy[status_] = batch_discount.actions_set
    elif status_ == param_list["n"]:
        batch_discount.policy[status_] = 1

# batch_discount.value_iteration(num_states=200)
batch_discount.policy_iteration(num_states=200)