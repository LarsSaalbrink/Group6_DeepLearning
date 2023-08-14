# %%
################### Imports ###################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt

# Download & preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

#Unsqueeze to add a dimension of size one
train_dataset.data = train_dataset.data.unsqueeze(-1)
test_dataset.data = test_dataset.data.unsqueeze(-1)

print("Training data size: ",train_dataset.data.shape)
print("Test data size: ",test_dataset.data.shape)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)

def train_model(model, num_epochs, HW_choice):
    print("CUDA available? ", torch.cuda.is_available())
    if HW_choice == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("HW choice: ", device)
    model.to(device)

    correct = 0
    total = 0
    val_loss_values = []
    loss_values = []
    accuracy_values = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} / {num_epochs}")
        loss_value = 0
        val_loss_value = 0
        model.train()

        # Training loop & loss
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            model.optimizer.zero_grad()
            outputs = model(data)
            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()
            loss_value += loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)

                # Validation loss
                val_loss = model.criterion(outputs, targets)
                val_loss_value += val_loss.item()

                # Accuracy
                _, predicted = torch.max(outputs.detach(), dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        loss_values.append(loss_value / len(train_loader))
        print(f"Training loss: {loss_value/len(train_loader):.4f}")

        val_loss_values.append(val_loss_value / len(test_loader))
        print(f"Validation loss: {val_loss_value/len(train_loader):.4f}")

        accuracy = correct / total
        accuracy_values.append(accuracy)
        print(f"Accuracy: {(100*accuracy):.2f}%")

    return loss_values, val_loss_values, accuracy_values


################### Plotting ###################
def plot(
    loss_values,
    val_loss_values,
    accuracy_values,
    num_epochs,
    filename="test_result.png",
):
    fig, ax1 = plt.subplots()

    # Set labels and ticks for the x-axis
    ax1.set_xlabel("Epochs")
    ax1.set_xticks(range(0, num_epochs))
    ax1.set_xticklabels(range(1, num_epochs + 1))
    ax1.set_ylabel("Loss")

    # Plot training and validation loss on the first y-axis (ax1)
    ax1.plot(loss_values, label="Training loss", color="blue")
    ax1.plot(val_loss_values, label="Validation loss", color="orange")

    # Create the second y-axis (ax2) for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(accuracy_values, label="Accuracy", color="green")

    # Add a point with its value displayed for the last accuracy data point
    last_accuracy = accuracy_values[-1]
    ax2.scatter(num_epochs - 1, last_accuracy, color="green", label=f"Last Accuracy: {last_accuracy*100:.3f}%")
 
    # Combine the legends from both y-axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper center")

    fig.savefig(filename)
    plt.show()
