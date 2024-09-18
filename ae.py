import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
epochs = 5
lr = 0.001
latent_dim = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root="/share/test/caohaidong/my_test",train=True,download=True,transform=transform)
test_dataset = datasets.CIFAR10(root="/share/test/caohaidong/my_test",train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    break

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64*4*4,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64*4*4),
            nn.ReLU(True),
            nn.Unflatten(1,(64,4,4)),
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,3,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder(latent_dim).to(device)
print("model shape:", model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = lr)

def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_index, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            if batch_index%10==0:
                print(f'Train epoch {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)}'
                      f'({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
train(model,device,train_loader,optimizer,epochs)

def test(model, device, test_loader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            loss = loss + criterion(output,data).item()
    loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(loss))
    
test(model,device,test_loader)

def display_reconstructed_images(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        output = model(data)
        
        n = min(data.size(0), 8)
        # 将数据从 GPU 移到 CPU
        data_cpu = data[:n].cpu()
        output_cpu = output[:n].cpu()
        
        # 归一化数据到 [0, 1] 范围
        data_normalized = torch.clamp(data_cpu, 0, 1)
        output_normalized = torch.clamp(output_cpu, 0, 1)
        
        fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(10, 4))
        pdb.set_trace()
        for i in range(n):
            ax = axes[0, i]
            ax.imshow(data_normalized[i].permute(1, 2, 0))
            ax.axis('off')
            
            # 展示重建图片
            ax = axes[1, i]
            ax.imshow(output_normalized[i].permute(1, 2, 0))
            ax.axis('off')
        plt.savefig("./ae_recon.png")


display_reconstructed_images(model, device, test_loader)
        