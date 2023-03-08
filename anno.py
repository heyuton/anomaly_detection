import torch
import torch.utils.data as Data

torch.manual_seed(1)  # reproducible

BATCH_SIZE = 8  # 每个batch的大小，取5或者8

# 生成测试数据
x = torch.linspace(0, 9, 10)  # x(torch tensor)
y = torch.linspace(9, 0, 10)  # y(torch tensor)

# 将输入和输出封装进Data.TensorDataset()类对象
torch_dataset = Data.TensorDataset(x, y)
print(type(torch_dataset))

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
    batch_size=BATCH_SIZE,  # 每块的大小
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多进程（multiprocess）来读数据
)

if __name__ == '__main__':  # 注意：如果loader中设置了num_workers!=0，即采用多进程来处理数据，运行含loader的操作必须在‘__main__’的范围内

    # 进行3轮训练（每次拿全部的数据进行训练）
    for epoch in range(3):
        # 在一轮中迭代获取每个batch（把全部的数据分成小块一块块的训练）
        for step, (batch_x, batch_y) in enumerate(loader):
            # 假设这里就是你训练的地方...

            # print出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x, '| batch y: ', batch_y)

'''
BATCH_SIZE = 5 时的结果
Epoch:  0 | Step:  0 | batch x:  tensor([4., 6., 9., 2., 3.]) | batch y:  tensor([5., 3., 0., 7., 6.])
Epoch:  0 | Step:  1 | batch x:  tensor([1., 0., 7., 8., 5.]) | batch y:  tensor([8., 9., 2., 1., 4.])
Epoch:  1 | Step:  0 | batch x:  tensor([3., 5., 6., 9., 7.]) | batch y:  tensor([6., 4., 3., 0., 2.])
Epoch:  1 | Step:  1 | batch x:  tensor([4., 2., 1., 0., 8.]) | batch y:  tensor([5., 7., 8., 9., 1.])
Epoch:  2 | Step:  0 | batch x:  tensor([3., 1., 4., 5., 9.]) | batch y:  tensor([6., 8., 5., 4., 0.])
Epoch:  2 | Step:  1 | batch x:  tensor([2., 8., 0., 7., 6.]) | batch y:  tensor([7., 1., 9., 2., 3.])
'''

# 当无法均等分成若干块时，先按每块BATCH_SIZE大小提取，最后剩下的不足BATCH_SIZE留作最后一块

'''
BATCH_SIZE = 8 时的结果 
Epoch:  0 | Step:  0 | batch x:  tensor([4., 6., 9., 2., 3., 1., 0., 7.]) | batch y:  tensor([5., 3., 0., 7., 6., 8., 9., 2.])
Epoch:  0 | Step:  1 | batch x:  tensor([8., 5.]) | batch y:  tensor([1., 4.])
Epoch:  1 | Step:  0 | batch x:  tensor([3., 5., 6., 9., 7., 4., 2., 1.]) | batch y:  tensor([6., 4., 3., 0., 2., 5., 7., 8.])
Epoch:  1 | Step:  1 | batch x:  tensor([0., 8.]) | batch y:  tensor([9., 1.])
Epoch:  2 | Step:  0 | batch x:  tensor([3., 1., 4., 5., 9., 2., 8., 0.]) | batch y:  tensor([6., 8., 5., 4., 0., 7., 1., 9.])
Epoch:  2 | Step:  1 | batch x:  tensor([7., 6.]) | batch y:  tensor([2., 3.])
'''

