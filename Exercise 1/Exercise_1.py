import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
import seaborn as sns 
import matplotlib.pyplot as plt  
# Download the MNIST dataset
transform = transforms.ToTensor() 
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the model
input_size = 64
hidden_size = 3
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(), nn.Linear(hidden_size, output_size), 
                      nn.LogSoftmax(dim=1))

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)




    



