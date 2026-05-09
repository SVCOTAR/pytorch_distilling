import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from torchvision.datasets import MNIST

#==设置随机种子
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark=True

train_dataset = MNIST(root='mnist_data/', train=True,download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='mnist_data/', train=False,download=True, transform=transforms.ToTensor())

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,pin_memory=(device.type == 'cuda'),num_workers=0,)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False,pin_memory=(device.type == 'cuda'),num_workers=0,)

class TeacherModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(TeacherModel,self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,1200)
        self.fc2=nn.Linear(1200,1200)
        self.fc3=nn.Linear(1200,num_classes)
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=x.view(-1,784)
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc2(x)
        x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc3(x)
        return x

#---1.teacher大模型的训练和评估
model=TeacherModel()
model=model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs=3
for epoch in range(epochs):
    model.train()
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #==前向预测
        preds=model(data)
        loss=criterion(preds,targets)
        
        #==反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Teacher] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))

teacher_model=model        

class StudentModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(StudentModel,self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,20)
        self.fc2=nn.Linear(20,20)
        self.fc3=nn.Linear(20,num_classes)
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=x.view(-1,784)
        x=self.fc1(x)
        #x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc2(x)
        #x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc3(x)
        return x


#----2.student简单模型的训练和评估
model=StudentModel()
model=model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs=3
for epoch in range(epochs):
    model.train()
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #==前向预测
        preds=model(data)
        loss=criterion(preds,targets)
        
        #==反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Student] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))
    
 
#---------------蒸馏teacher大模型

#准备预训练好的教师模型
teacher_model.eval()

#准备新的学生模型
model=StudentModel()
model=model.to(device)
model.train()

temp= 4


#hard_loss
hard_loss=nn.CrossEntropyLoss()
#hard_loss 权重
alpha= 0.3

#soft_loss 
soft_loss=nn.KLDivLoss(reduction="batchmean")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


#--3. 蒸馏模型训练和评估
epochs=3
for epoch in range(epochs):
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #--techer 模型
        with torch.no_grad():
            teacher_preds=teacher_model(data)
        
        #--学生模型
        student_preds = model(data)
        
        #--计算 hard_loss
        student_loss=hard_loss(student_preds,targets)
        
        #--计算蒸馏后的预测结果及soft_loss
        distillation_loss=soft_loss(F.log_softmax(student_preds/temp,dim=1),F.softmax(teacher_preds/temp,dim=1))* (temp ** 2)
        
        #将hard_loss和soft_loss加权求和
        loss=alpha*distillation_loss+(1-alpha)*student_loss
        
        #方向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Distill] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))
