# 生成数据集

# In[58]:

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# In[59]:

num_inputs=2
num_examples=1000
true_w=torch.tensor([[2.0,-3.4]]).t().to(torch.float32)
true_b=4.2
features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs))).to(torch.float32)

# In[60]:

labels=torch.mm(features,true_w)+true_b
labels+=torch.from_numpy(np.random.normal(0,0.1,size=labels.size()))

# In[61]:

print(features[0], labels[0])

# In[62]:

plt.scatter(features[:,1].numpy(), labels.numpy(),1)


# 读取数据

# In[63]:

#遍历数据集，并不断读取小批量数据样本，知道数据集全部使用
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)##
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size,num_examples)]) # 最后⼀次可能不⾜⼀个batch
        #LongTensor：float64
        yield features.index_select(0, j), labels.index_select(0, j)

# In[64]:

batch_size=10
print(data_iter(batch_size,features,labels))
i=0
for X,y in data_iter(batch_size,features,labels):
    i=i+1
print(X,y)
print('一共获取了{}份小样本'.format(i))

# 初始化模型参数

# In[65]:

w=torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b=torch.zeros(1,dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
print(w,b)

# 定义模型

# In[66]:

#定义模型
def linreg(X,w,b):
    return torch.mm(X,w)+b
#损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2
#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param.data-=lr*param.grad/batch_size#注意param.data
        

# 训练模型

# In[67]:

lr=0.03
num_epochs=10
net=linreg
loss=squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l=loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)
        
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l=loss(net(features,w,b),labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# In[68]:

print(true_w, '\n', w)
print(true_b, '\n', b)

