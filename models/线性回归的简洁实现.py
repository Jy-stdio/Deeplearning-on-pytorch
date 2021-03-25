# # 线性回归的简洁实现

# In[345]:


import torch.nn as nn


# In[346]:


#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples,num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] +true_b
labels += torch.tensor(np.random.normal(0, 0.01,size=labels.size()), dtype=torch.float)
print('第一个样本的特征为{},标签为{}'.format(features[0,:],labels[0]))


# In[347]:


import torch.utils.data as Data

batch_size=10
#将训练数据的特征和标签组合
dataset=Data.TensorDataset(features,labels)
#随机读取小批量
data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)


# In[348]:


#读取并打印第一个小批量数据样本
for X,y in data_iter:
    print("第一个小批量数据样本如下：")
    print('特征{}，\n标签{}'.format(X,y))
    break


# 定义模型

# In[349]:


##网络搭建方式1:nn.Module
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(n_feature,1)
    #forward定义前向传播
    def forward(self,x):
        y=self.linear(x)
        return y
net=LinearNet(num_inputs)
print(net)


# In[350]:


#网络搭建方式2：nn.Sequential
net2=nn.Sequential(nn.Linear(num_inputs,1))
print(net2)
print(net2[0])


# In[351]:


#查看模型的参数
for param in net2.parameters():
    print(param)
#注意： torch.nn 仅⽀持输⼊⼀个batch的样本不⽀持单个样本输⼊，如果只有单个样本，可使
#⽤ input.unsqueeze(0) 来添加⼀维。


# In[352]:


#初始化模型参数
from torch.nn import init

init.normal_(net2[0].weight,mean=0,std=0.01)
init.constant_(net2[0].bias,val=0)
print(net2[0].weight,net2[0].bias)


# In[353]:


#损失函数
loss=nn.MSELoss()


# In[354]:


import torch.optim as optim

optimizer=optim.SGD(net2.parameters(),lr=0.03)
print(optimizer)


# In[355]:


# #我们还可以为不同⼦⽹络设置不同的学习率，这在finetune时经常⽤到。
# optimizer =optim.SGD([
#  # 如果对某个参数不指定学习率，就使⽤最外层的默认学习率
#  {'params': net.subnet1.parameters()}, # lr=0.03
#  {'params': net.subnet2.parameters(), 'lr': 0.01}
#  ], lr=0.03)


# In[356]:


# # 调整学习率
# for param_group in optimizer.param_groups:
#  param_group['lr'] *= 0.1 # 学习率为之前的0.1倍


# In[357]:


#训练模型
num_epochs=10
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output=net2(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


# In[358]:


#⽐较学到的模型参数和真实的模型参数
dense = net2[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
