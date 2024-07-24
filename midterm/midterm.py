from typing import List  # Corrected import statement
from tqdm import tqdm
from torchvision import models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch import nn
from torchsummary import summary
from time import time

batch_size = 16
num_workers = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))] # 이미지넷 pretrained했기 때문
)



train_loader = DataLoader(
        CIFAR10(root=r'C:\Users\choi\Desktop\fluid_detection\meanterm_data', train=True, download=False, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )


test_loader = DataLoader(
        CIFAR10(root=r'C:\Users\choi\Desktop\fluid_detection\meanterm_data', train=False, download=False, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = models.efficientnet_v2_s(pretrained=True)

# import pdb; pdb.set_trace()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 10)

model = model.cuda()


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 30
model.train()

# summary(model, (3, 224, 224))   # 구조 확인
# import pdb; pdb.set_trace()

start_total = time()
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    epoch_acc = 0
    for x, y in tqdm(train_loader):
        start_epoch = time()
        x = torch.FloatTensor(x).cuda()
        y = torch.FloatTensor(y).cuda()

        out = model(x)
        cost = loss(out, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        hypothesis = torch.nn.functional.softmax(out, dim=1)
        pred = torch.argmax(hypothesis, dim=1)

        correct_pred = (pred == y)

        epoch_loss += cost.item()
        epoch_acc += correct_pred.sum()
        print("train time : {}sec".format(int(time()-start_epoch)))
        filename = 'C:/Users/wnstj/OneDrive/바탕 화면/세종대/4-1/딥시/midterm/checkpoint/chekpoint' + str(epoch + 1) + 'size_224_basic.tar'
        torch.save(model.state_dict(), filename)
        import pdb; pdb.set_trace()

    if epoch%10 == 0:
        print("epoch : {} loss : {} acc : {}".format(epoch+1, epoch_loss, epoch_acc/len(train_loader.dataset)))
        filename = 'C:/Users/wnstj/OneDrive/바탕 화면/세종대/4-1/딥시/midterm/checkpoint/chekpoint' + str(epoch + 1) + 'size_224_basic.tar'
        torch.save(model.state_dict(), filename)

    
print("train time : {}sec".format(int(time()-start_total)))

model.eval()
acc = 0
with torch.no_grad():
    for x, y in test_loader:
        x = torch.FloatTensor(x).cuda()
        y = torch.FloatTensor(y).cuda()
        
        out = model(x)

        hypothesis = torch.nn.functional.softmax(out, dim=1)
        pred = torch.argmax(hypothesis, dim=1)

        correct_pred = (pred == y)

        acc += correct_pred.sum()

    print("acc : {}".format(acc/len(test_loader.dataset)))
