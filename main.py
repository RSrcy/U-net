import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from U_Net_2 import Ren
from MyDataset import MyDataset

#
root_dir = "E:/test3/a"
Train_MyData = MyDataset(root_dir)
root_dir_2 = "E:/test3/b"

Test_MyData = MyDataset(root_dir_2)
writer = SummaryWriter('logs')
# 利用Dataloader来加载数据集
Train_loaders = DataLoader(dataset=Train_MyData, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
Test_loaders = DataLoader(dataset=Test_MyData, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
Test_data_size = len(Test_loaders)
print(type(Train_loaders))
# 创建网络模型
Ren = Ren()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.RMSprop(Ren.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试次数
epoch = 50  # 训练次数

for i in range(epoch):
    print("---------------第{}轮训练开始----------------".format(i + 1))
    # 训练步骤开始
    Ren.train()
    for data in Train_loaders:
        imgs, targets = data
        outputs = Ren(imgs)
        # print(outputs.shape)
        # print(targets.shape)
        # targets = targets.squeeze(0)
        # print(targets.shape)
        targets = targets / 1.0
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        print("训练次数:{}, Loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar('loss', loss.item(), total_train_step)
    # 测试步骤开始
    Ren.eval()
    total_accuracy = 0  # 记录训练正确率
    with torch.no_grad():
        for data in Test_loaders:
            imgs, targets = data
            outputs = Ren(imgs)
            targets = targets / 1.0
            loss_2 = loss_fn(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            total_test_step = total_test_step + 1
            print("测试次数:{}, Loss:{}".format(total_test_step, loss_2.item()))
            writer.add_scalar('loss', loss_2.item(), total_test_step)
            print("此次正确率:{}".format(accuracy))
            writer.add_scalar('test_accuracy', total_accuracy/Test_data_size, total_test_step)
    torch.save(Ren, "Ren_{}.pth".format(i))
    print("over")
writer.close()
