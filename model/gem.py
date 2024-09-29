# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog

from model.common import MLP, ResNet18
#from common import MLP, ResNet18

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):#  定义函数compute_offsets，用于计算偏移量
    """
        Compute offsets for cifar to determine which计算cifar的偏移量以确定为给定任务选择的输出。
        outputs to select for a given task.
    """
    if is_cifar:         #  如果是  CIFAR  数据集，则计算偏移量
        offset1 = task * nc_per_task     #  第一个偏移量为任务号乘以任务中每个类别的数量
        offset2 = (task + 1) * nc_per_task   #  第二个偏移量是下一个任务的第一个类别的位置
    else:   #  如果不是  CIFAR  数据集，则偏移量为  0  和每个任务的类别数量
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2   #  返回偏移量1和偏移量2


def store_grad(pp, grads, grad_dims, tid):  #  定义一个函数用于存储过去任务的参数梯度
    """
        This stores parameter gradients of past tasks.这个函数用于存储过去任务的参数梯度。
        pp: parameters pp  是参数，grads  是梯度，grad_dims  是每层参数数量的列表，tid  是任务编号
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)   #  将  grads  对应列  tid  中的所有值设置为  0.0
    cnt = 0    #  初始化计数器  cnt  为  0
    for param in pp():    #  遍历参数  pp()  中的所有参数
        if param.grad is not None:  #  如果参数的梯度不为  None
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])   #  如果是第一个参数，则  beg  为  0；否则  beg  为前面所有参数的梯度维度之和
            en = sum(grad_dims[:cnt + 1])   #  en  为前  cnt  +  1  个参数的梯度维度之和
            grads[beg: en, tid].copy_(param.grad.data.view(-1))  #  将参数  param  的梯度数据展平后复制到对应行列中
        cnt += 1   #  cnt    =  1    #  计数器加1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient  这用于用新的渐变覆盖渐变矢量，无论何时发生违规。
pp:参数newgrad：校正的坡度 grad_dims：存储每层参数数量的列表
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0  #  初始化计数器为0
    for param in pp():  #  遍历pp()函数返回的参数列表，即遍历模型的所有参数
        if param.grad is not None:  #  如果梯度不为  None，则存储参数梯度   #  判断当前参数是否存在梯度  如果存在梯度，则存储参数梯度
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])  #  根据计数器cnt计算当前参数梯度在newgrad中的位置 beg的值将根据cnt的值进行计算。如果cnt等于0，则beg的值为0；否则，beg的值为grad_dims列表中前cnt个元素的总和。
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size()) #  从newgrad中提取出当前参数对应的梯度，并重新排列形状
            param.grad.data.copy_(this_grad) #  然后将提取的梯度复制到当前参数的grad属性上
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        # self.is_cifar = (args.data_file == 'cifar100.pt')
        # if self.is_cifar:
        #     self.net = ResNet18(n_outputs)
        # else:
        #     self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.data_file = args.data_file
        self.is_cifar = (
                    args.data_file == 'cifar100.pt' or args.data_file == "cifar10.pt" or "mini_imagenet" in args.data_file)
        # self.is_cifar = (args.data_file == 'cifar10.pt')  # 判断是否为cifar100数据集

        #self.state = {}  # 添加存储模型参数状态的字典
        if self.is_cifar:  # 如果是cifar100数据集
            self.net = ResNet18(n_outputs, args.data_file)  # 神经网络为ResNet18    ********
        else:  # 如果不是cifar100数据集
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])  # 神经网络为MLP  ********
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
        self.opt.step()
