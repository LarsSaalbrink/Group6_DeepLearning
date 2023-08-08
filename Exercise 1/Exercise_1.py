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

################### Part 1 ###################
input_size = 28*28  #Dataset contains 28x28 images
hidden_sizes = [128, 64]
output_size = 10
epochs = 1

### a ###
### b ###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Linear is a fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

        #Works MUCH better than sigmoid & tanh, 
        #slightly better than regular softmax
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x

### c ###
model = Net()
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

### d ###
#Training loop
running_loss_arr = []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = loss_func(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(train_loader)}")
        running_loss_arr.append((running_loss/len(train_loader)))

### e ###
################### Part 2 ###################
#Eval loop
test_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        log_ps = model(images)
        test_loss += loss_func(log_ps, labels)
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class==labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    else:
        print(f"Test loss: {test_loss/len(test_loader)}")
        print(f"Accuracy: {accuracy/len(test_loader)}")  #Model accuracy

### f ###
torch.save(model.state_dict(), 'checkpoint.pth')

################### Part 3 ###################
#Loss plot
plt.figure(figsize=(12,6))
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1,epochs))
plt.plot(running_loss_arr, label="Training Loss")
# plt.plot(test_loss/len(test_loader), label="Test Loss")
plt.show()


#Run the first item in the dataset through the model and print the output
#(Not part of the exercises)
images, labels = next(iter(test_loader))
img = images[0].view(1, 28*28)
with torch.no_grad():
    log_ps = model(img)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    print(f"Predicted digit: {top_class.item()}")

#Plot the first item in the dataset
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
plt.show()



