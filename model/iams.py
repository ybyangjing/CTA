import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import quadprog
from utils import *

from model.common import MLP, ResNet18  # 导入自己创建的MLP和ResNet18模型

# Import GradCAM class  #  导入GradCAM类
from pytorch_grad_cam import GradCAM

#  设置设备为cuda或cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#  开启自动求导的异常检测
torch.autograd.set_detect_anomaly(True)


# Auxiliary functions useful for GEM's inner optimization.
#  定义函数，计算CIFAR的偏移量来确定选择哪些输出
def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.计算cifar的偏移量以确定为给定任务选择的输出。
    """
    if is_cifar:  # 如果是CIFAR
        offset1 = task * nc_per_task  # 计算偏移量
        offset2 = (task + 1) * nc_per_task
    else:  # 否则偏移量均为0
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2  # 返回偏移量


def store_grad(pp, grads, grad_dims, tid):  # 定义函数，存储过去任务的参数梯度
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients    将该任务的梯度全部置为0
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():  # 遍历每个参数
        if param.grad is not None:  # 如果参数梯度不为空
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])  # 计算该参数的梯度在grads中的起始位置和终止位置
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))  # 将该参数的梯度复制到grads中对应位置
        cnt += 1  # 计数器加1


def overwrite_grad(pp, newgrad, grad_dims):  # 定义了一个函数，用于将梯度更新为新的梯度
    """
        This is used to overwrite the gradients with a new gradient   这个函数用于在发生违规时，用新的梯度向量覆盖原有的梯度。
        vector, whenever violations occur.
        pp: parameters       pp:  参数
        newgrad: corrected gradient   newgrad:  纠正后的梯度
        grad_dims: list storing number of parameters at each layer     grad_dims:  存储每个层的参数数目的列表
    """
    cnt = 0  # 统计层数的计数器
    for param in pp():  # 遍历每一个参数
        if param.grad is not None:  # 如果梯度不为空
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])  # 按照层数获取该层参数在梯度中的起始位置
            en = sum(grad_dims[:cnt + 1])  # 按照层数获取该层参数在梯度中的结束位置
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())  # 获取该层参数在新梯度中的具体数值，然后用这个数值更新原有梯度
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):  # 定义了一个函数，解决了GEM双重QP问题
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        给定一个拟议的梯度“gradient”和任务梯度“memories”，并解决了论文中描述的GEM双重QP问题。  将“gradient”重写为最终的投影更新。
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()  # 将GPU的张量转换为CPU的numpy数组，然后进行转置，然后转换为double类型的数组
    gradient_np = gradient.cpu().contiguous().view(
        -1).double().numpy()  # 将GPU的张量转换为CPU的numpy数组，并进行内存连续性处理，然后将其视为一维数组，并将其转换为double类型的数组
    t = memories_np.shape[0]  # 获取任务梯度中任务的数量
    P = np.dot(memories_np, memories_np.transpose())  # 计算内积矩阵P
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps  # 对P进行修正
    q = np.dot(memories_np, gradient_np) * -1  # 计算向量q
    G = np.eye(t)  # 设置单位矩阵G
    h = np.zeros(t) + margin  # 设置向量h
    v = quadprog.solve_qp(P, q, G, h)[0]  # 使用QP求解器解决双重QP问题
    x = np.dot(v, memories_np) + gradient_np  # 计算x
    gradient.copy_(torch.Tensor(x).view(-1, 1))  # 将x转换为Tensor，并使用其更新原有梯度张量


class Net(nn.Module):  # 定义了一个神经网络的类Net
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args,
                 mas_coef=0.3, l2_coef=0.005):  # mas_coef=0.005
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength  # 设定记忆强度，即存储新数据的程度
        self.is_cifar = (args.data_file == 'cifar10.pt')  # 判断是否为cifar100数据集
        self.mas_coef = mas_coef  # 添加新的超参数 mas_coef
        self.l2_coef = l2_coef
        self.state = {}  # 添加存储模型参数状态的字典
        if self.is_cifar:  # 如果是cifar100数据集
            self.net = ResNet18(n_outputs)  # 神经网络为ResNet18    ********
            self.target_layer = self.net.layer4[-1]  # 设定目标层，即最后一层
            self.cam = GradCAM(model=self.net, target_layer=self.target_layer,
                               use_cuda=True)  # 实例化GradCAM，使用cuda  **********
        else:  # 如果不是cifar100数据集
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])  # 神经网络为MLP  ********
        #  以上为初始化神经网络所需运行的代码
        self.ce = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        self.n_inputs = n_inputs  # 设定输入维度
        self.n_outputs = n_outputs  # 设定输出维度

        self.opt = optim.SGD(self.parameters(), args.lr)  # 定义优化器

        self.n_memories = args.n_memories  # 设定记忆库数量
        self.gpu = args.cuda  # 判断是否使用GPU

        # allocate episodic memory  分配情节记忆
        self.memory_data = {}  # 初始化记忆数据
        self.memory_labs = {}  # 初始化记忆标签
        self.pxl_needed = {}  # 初始化像素
        self.importance = {}
        # self.importance = np.ones((n_tasks, n_outputs))
        # 对记忆数据和记忆标签进行初始化
        # self.memory_data = torch.FloatTensor(
        #     n_tasks, self.n_memories, n_inputs)
        # self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        # if args.cuda:
        #     self.memory_data = self.memory_data.cuda()
        #     self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory+
        self.grad_dims = []  # 初始化梯度维度
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []  # 初始化观察到的任务
        self.old_task = -1  # 初始化老任务
        self.mem_cnt = 0  # 初始化记忆计数器
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # set the threshold for GradCAM mast
        self.theta = args.theta  # 设定GradCAM门的阈值

        # set parameters for dynamic memory
        self.max_pxl = self.n_memories * self.n_inputs  # 设定最大像素
        self.pxl_stored = np.zeros(n_tasks)  # 初始化存储的像素
        self.img_stored = np.zeros(n_tasks)  # 初始化存储的图像像素

        # MAS相关代码，用于提取Memory重要参数
        self.omega = []
        for param in self.parameters():
            self.omega.append(torch.zeros_like(param.data, requires_grad=False))
        self.prev_param = None

    # *************************************
    # def compute_importance(self, x, t):
    #     bs = x.size(0)
    #     importance = []
    #     for param in self.parameters():
    #         importance.append(torch.zeros_like(param))
    #     losses = torch.zeros(bs)
    #
    #     for i in range(bs):
    #         output = self.net(x[i].unsqueeze(0))
    #         offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
    #         t_tensor = torch.tensor(t)
    #         loss = self.ce(output[:, offset1:offset2], t_tensor)
    #         #loss = self.ce(output[:, offset1:offset2], t)
    #         losses[i] = loss.item()
    #         loss.backward()
    #         for j, param in enumerate(self.parameters()):
    #             importance[j] += torch.abs(param.grad.data)
    #
    #     total_loss = torch.sum(losses)
    #     for j, param in enumerate(self.parameters()):
    #         importance[j] *= losses / total_loss
    #
    #     return importance

    def update_parameters(self, x, t, y):
        self.zero_grad()
        output = self.net(x)
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(output[:, offset1:offset2], y - offset1)

        if t > 0:
            prev_output = self.forward(x, t - 1)
            prev_offset1, prev_offset2 = compute_offsets(t - 1, self.nc_per_task, self.is_cifar)
            regularization_loss = F.mse_loss(output[:, prev_offset1:prev_offset2],
                                             prev_output[:, prev_offset1:prev_offset2])
            loss += regularization_loss
        loss.backward()

        if self.mas_coef > 0:
            for ln in self.net._modules.values():
                for p in ln.parameters():
                    param_state = self.state.get(p, {})
                    prev_grad = param_state.get('grad', torch.zeros_like(p))
                    hessian = (prev_grad - p.grad) / self.margin
                    param_state['grad'] = p.grad.clone()
                    self.state[p] = param_state
                    p.grad += self.mas_coef * hessian

        self.opt.step()

    def compute_importance(self):
        importances = []
        for param in self.parameters():
            importance = torch.norm(param.grad) / torch.norm(param)
            importances.append(importance.item())
        max_importance = max(importances)
        importances = [importance / max_importance for importance in importances]
        return importances

    def loss_with_importance(self, outputs, targets, importances, alpha=0.1):
        ce_loss = self.ce(outputs, targets)
        penalty = 0
        for param, importance in zip(self.parameters(), importances):
            penalty += torch.norm(param) * importance
        total_loss = ce_loss + alpha * penalty
        return total_loss

    # **************************************

    def forward(self, x, t):  # 定义类，名称为forward
        # output = self.net(x) * torch.from_numpy(self.importance[t, :]).float().cuda().unsqueeze(0).expand(x.size(0), -1)
        output = self.net(x)  # 将x作为输入传递给网络net，并将其结果存储在output中
        if self.is_cifar:  # 如果是CIFAR数据集
            # make sure we predict classes within the current task #  保证我们预测当前任务内的类别
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:  # 如果offset1大于0，则用-10e10填充output的前offset1列
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:  # 如果offset2小于总输出数，则用-10e10填充output的从offset2开始的列
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):  # 定义类，名称为observe

        # mini-batch size
        bsz = y.data.size(0)  # 获取y的大小，并将其作为batch  size

        # new task comes  #  如果有新的任务出现
        if t != self.old_task:
            self.observed_tasks.append(
                t)  # 将新任务添加到observed_tasks列表中   先判断新任务t是否与上一个任务（old_task）相同，如果不同则将t添加到observed_tasks中，并更新old_task为t。
            self.old_task = t  # 更新old_task
            # initialize episodic memory for the new task  #  为新任务初始化记忆体
            self.memory_data[t] = torch.FloatTensor(bsz,
                                                    self.n_inputs)  # 分别为新任务t分配一些空间，包括FloatTensor类型的memory_data（用于存储样本特征）、LongTensor类型的memory_labs（用于存储样本标签）以及一个大小为bsz的数组pxl_needed（表示该任务需要的像素数）。
            self.memory_labs[t] = torch.LongTensor(bsz)
            self.pxl_needed[t] = np.zeros(bsz)
            if self.gpu:  # 如果使用GPU，则将数据和标签移到GPU上  如果使用GPU，则将memory_data和memory_labs移动到GPU上
                self.memory_data[t].cuda()
                self.memory_labs[t].cuda()

            total_importance = np.zeros(len(self.omega))

            # for i in range(self.n_inputs):
            for i in range(len(self.omega)):
                total_importance[i] += np.sum(self.omega[i].cpu().data.numpy())
            total_importance /= np.sum(total_importance)
            print('total importance:', total_importance)

        # if self.mem_cnt > 0:
        #     self.project_importance(t, bsz)

        # compute gradient on previous tasks      #  计算前一个任务的梯度
        if len(self.observed_tasks) > 1:  # 代码首先判断模型已经观察到的任务数量是否大于1，如果是则进入for循环。
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()  # 将梯度清零 可避免历史梯度对当前训练的影响。
                # fwd/bwd on the examples in the memory  #  对记忆中的示例进行前向传播和反向传播
                past_task = self.observed_tasks[tt]  # 在循环中，代码将当前的梯度清零，然后选择一个之前的任务past_task进行前向传播和反向传播。

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)  # 在前向传播中，代码将past_task对应的记忆数据输入模型，再对offset1到offset2范围内的输出进行损失计算。  函数能够计算偏移量并返回offset1和offset2。
                #  对输入样本进行前向传播并计算损失值。
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task], past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                ptloss.backward()  # 在反向传播中，代码将这个损失值反向传播回模型，并将梯度存储在grads列表中。  这样，代码就完成了对历史训练任务的反向传播，并将这些任务对应的梯度存储在grads列表中，以便后续在共享层中使用。#  将这个损失值反向传播回模型，并将梯度存储在  grads  列表中。
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)  # store_grad  函数能够将参数和梯度大小存储到  grads  列表中。

        self.update_parameters(x, t, y)

        # # 调用 compute_importance 方法
        # importance = self.compute_importance(x, t)
        # print(importance)  # 打印重要性计算结果
        #
        # # 调用 parameter_norm_penalty 方法
        # penalty = self.parameter_norm_penalty()

        #
        # # self.project_importance(task=t, mem_batch_size=256)

        # now compute the grad on the current minibatch
        # self.zero_grad() #  设置所有梯度为0

        # logits = self.forward(x, t)
        # loss = self.ce(logits, y)
        # loss.backward()
        # store_grad(self.parameters, self.grads, self.grad_dims, t)

        if t not in self.memory_data:
            inputs_cpu = x.detach().clone().cpu()
            labs_cpu = y.detach().clone().cpu() - compute_offsets(t, self.nc_per_task, self.is_cifar)[0]
            self.img_stored[t] += inputs_cpu.size(0)
            amount_to_save = min(inputs_cpu.size(0), int(self.max_pxl / self.n_inputs))

            for i in range(inputs_cpu.size(0)):
                if self.pxl_needed[t][i] == 0:
                    img = inputs_cpu[i].view(-1).numpy()
                    self.memory_data[t][self.pxl_stored[t]] = torch.from_numpy(img)
                    self.memory_labs[t][self.pxl_stored[t]] = labs_cpu[i]
                    self.pxl_stored[t] += 1

                    if self.pxl_stored[t] >= self.max_pxl / self.n_inputs:
                        self.pxl_needed[t][:] = 1
                        break

        offset1, offset2 = compute_offsets(t, self.nc_per_task,
                                           self.is_cifar)  # 根据任务t、每个任务的类别数self.nc_per_task、是否CIFAR数据集self.is_cifar计算偏移量
        # loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)     #按照偏移量，将输入x送到网络中进行前向计算，计算损失loss

        importances = self.compute_importance()
        loss = self.loss_with_importance(self.forward(x, t)[:, offset1:offset2], y - offset1, importances)
        # l2_loss = 0.0
        # for param in self.parameters():
        #     l2_loss += torch.norm(param)
        # loss += self.l2_coef * l2_loss
        #
        # loss.backward()   #  对损失进行反向传播，计算梯度
        # 计算参数范数惩罚（权重衰减）
        # reg_loss = None
        reg_loss = 0.0
        for param in self.parameters():
            if reg_loss is None:
                reg_loss = torch.norm(param, p='fro')  # 使用Frobenius范数作为正则化项
            else:
                reg_loss += torch.norm(param, p='fro')
        loss += 0.02 * reg_loss  # 将正则化损失添加到总体损失中
        loss_channels = []

        loss.backward()

        # if self.mas_coef > 0:
        #     for ln in self.net._modules.values():
        #         for p in ln.parameters():
        #             # 获取梯度信息，并计算二阶导数
        #             param_state = self.state.get(p, {})
        #             prev_grad = param_state.get('grad', torch.zeros_like(p))
        #             hessian = (prev_grad - p.grad) / self.margin
        #             # 更新梯度信息
        #             param_state['grad'] = p.grad.clone()
        #             self.state[p] = param_state
        #             # 添加正则化
        #             p.grad += self.mas_coef * hessian
        #
        # self.opt.step()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:  # 如果观察到的任务数量大于1，执行以下操作：
            # copy gradient  复制梯度
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])  # 获取除最后一个任务以外的所有任务的索引
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),  # 计算当前任务的梯度和其他任务的梯度之间的点积
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:  # 如果点积小于0，则将当前任务的梯度投影到锥体中
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back        #  将梯度复制回去
                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)

                # MAS相关代码，更新重要度参数
                param_list = list(self.parameters())
                if self.prev_param is None:
                    self.prev_param = []
                    for param, omega in zip(param_list, self.omega):
                        self.prev_param.append(param.data.clone())
                        omega += torch.abs(param.grad.data)
                else:
                    for param, omega, prev_param in zip(param_list, self.omega, self.prev_param):
                        omega += torch.abs(param.grad.data * (param.data - prev_param))
                        prev_param.copy_(param.data)
        self.opt.step()  # 执行优化器的反向传播

        # Update ring buffer storing examples from current task with memory efficiency by GradCAM  使用  GradCAM  更新存储当前任务示例的环形缓冲区，以达到节省内存的效果
        tmp_x_data = x.data  # tensor shape: bsz by 3*32*32  获取输入  tensor，该  tensor  的形状为  bsz*3*32*32
        tmp_x_data = tmp_x_data.view(tmp_x_data.size(0), 3, 32,
                                     32)  # convert the shape to be 4D  #  将  tensor  的形状转换为  4D
        original_x = tmp_x_data.clone()  # 克隆一个新的  tensor，并将其转换为  np.float32  类型
        original_x = np.float32(original_x.detach().cpu())

        target_category = None  # y.detach().cpu().tolist()#  目标类别设为  None
        grayscale_cam = self.cam(input_tensor=tmp_x_data, target_category=target_category, task_index=t)  # 计算  GradCAM
        masked_x = torch.empty_like(tmp_x_data)  # 生成一个空的  torch  tensor  用于存储掩码图像
        pxl_needed = np.zeros(
            bsz)  # number of non-zero pixels for each image within this mini-batch  #  为这个  mini-batch  中的每个图像计算非零像素的数量

        tmp_x_data = tmp_x_data * 255.0  # convert image back to 0 - 255 value range将图像重新转换为0-255的数值范围
        for i in range(bsz):
            tmp_x = np.uint8(tmp_x_data[i].detach().cpu())  # 将tensor转换为numpy.uint8
            a = original_x[i, :]  # 取出原始图像
            a = np.rollaxis(a, 0, 3)  # 将通道维度从第一维移到最后一维
            tmp_x = np.rollaxis(tmp_x, 0, 3)
            tmp_gc = grayscale_cam[i, :]  # 获取  GradCAM  结果
            # get GradCAM mask by threshold theta  #  根据阈值theta获取GradCAM掩码
            mask = np.where(tmp_gc < self.theta, 1, 0)
            # calculate number of non-zero pixels of this image after applying the mask  计算应用掩码后该图像中非零像素的数量
            pxl_needed[i] = 3 * 32 * 32 - 3 * np.count_nonzero(mask)
            # mask the image 应用掩码到图像上
            mask = np.uint8(mask)  # 将掩码应用到图像上
            tmp_inpainted = cv2.inpaint(tmp_x, mask, 3, cv2.INPAINT_TELEA)  # 使用  cv2.inpaint  创建掩膜图像
            tmp_inpainted = tmp_inpainted / 255.0  # 转为  0-1  数据值范围
            tmp_inpainted = np.rollaxis(tmp_inpainted, 2, 0)  ##  将通道维度从最后一个维度移到第一个维度
            masked_x[i] = torch.from_numpy(tmp_inpainted).to(masked_x)  # 将  inpainted  图像添加到  masked_x

        # get the mini-batch data after GradCAM  获取GradCAM后的mini-batch（批次数据）
        masked_x = masked_x.view(masked_x.size(0), -1)  # 获取  GradCAM  后的  mini-batch
        masked_x.cuda()

        total_pxl_needed = np.sum(pxl_needed)  # 计算所有图片应用掩码后，非零像素的数量之和。

        # now we begin to store the mini-batch into episodic memory with dynamic size 开始将mini-batch动态存储到记忆库中
        if self.img_stored[t] == 0:  # 如果该时刻的图像存储量为0
            self.memory_data[t].copy_(masked_x)  # 将masked_x复制到memory_data[t]
            self.img_stored[t] += bsz  # 更新该时刻的图像存储量
            self.pxl_stored[t] += total_pxl_needed  # 更新该时刻所需像素总数
            self.memory_labs[t].copy_(y)  # 将y复制到memory_labs[t]
            self.pxl_needed[t] = pxl_needed  # 更新该时刻的像素需求量

        elif self.pxl_stored[t] + total_pxl_needed <= self.max_pxl:  # 如果该时刻的像素存储量加上当前mini-batch的像素需求量不超过最大像素存储量
            self.memory_data[t] = torch.cat((self.memory_data[t].cuda(), masked_x),
                                            0)  # 将当前mini-batch的masked_x合并到memory_data[t]
            self.img_stored[t] += bsz  # 更新该时刻的图像存储量
            self.pxl_stored[t] += total_pxl_needed  # 更新该时刻所需像素总数
            self.memory_labs[t] = torch.cat((self.memory_labs[t].cuda(), y))  # 将当前mini-batch的y合并到memory_labs[t]
            self.pxl_needed[t] = np.concatenate((self.pxl_needed[t], pxl_needed),
                                                axis=None)  # 将当前mini-batch的像素需求量合并到pxl_needed[t]

        else:  # 如果该时刻的像素存储量加上当前mini-batch的像素需求量超过最大像素存储量
            pxl_released = 0
            for k in range(int(self.img_stored[t])):  # 遍历该时刻的所有图像，找到第一个能够容纳当前mini-batch的位置
                pxl_released += self.pxl_needed[t][k]
                if self.pxl_stored[t] + total_pxl_needed - pxl_released <= self.max_pxl:
                    # remove images up to the current one from memory从存储库中移除当前图像之前的图像
                    self.memory_data[t] = self.memory_data[t][k + 1:, ]
                    self.memory_labs[t] = self.memory_labs[t][k + 1:]
                    self.pxl_needed[t] = self.pxl_needed[t][k + 1:]
                    self.img_stored[t] -= k + 1  # 更新该时刻的图像存储量
                    self.pxl_stored[t] -= pxl_released  # 更新该时刻的像素存储量
                    # now store the current mini-batch into memory  将当前mini-batch存储到存储库中
                    self.memory_data[t] = torch.cat((self.memory_data[t].cuda(), masked_x),
                                                    0)  # 将当前mini-batch的masked_x合并到memory_data[t]
                    self.img_stored[t] += bsz  # 更新该时刻的图像存储量
                    self.pxl_stored[t] += total_pxl_needed  # 更新该时刻的像素存储量
                    self.memory_labs[t] = torch.cat((self.memory_labs[t].cuda(), y))  # 将当前mini-batch的y合并到memory_labs[t]
                    self.pxl_needed[t] = np.concatenate((self.pxl_needed[t], pxl_needed),
                                                        axis=None)  # 将当前mini-batch的像素需求量合并到pxl_needed[t]
                    break
                else:
                    continue



