import numpy as np

################### Part 1 ###################
### a ###
a = np.full((2, 3), 4)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.eye(2, 3)
d = a + b + c

print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)

### b ###
a = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [6, 7, 8, 9, 0], [0, 9, 8, 7, 6]])

sum = np.sum(a, axis=0)
print("\n\nSum: ", sum)

transpose = np.transpose(a)
print("\n\nTranspose: ", transpose)


################### Part 2 ###################
import pandas as pd

### b ###
df = pd.read_csv('auto.csv')

### c ###
for row in df.itertuples():
    if row.mpg < 16:
        df = df.drop(row.Index)

### d ###
print("\n\nweight, acceleration")
for row in df.head(7).itertuples():
    print(row.weight, row.acceleration)

### e ###
#Remove the rows in the ‘horsepower’ column that has the value ‘?’, and
#convert the column to an ‘int’ type instead of a ‘string’
df = df[df.horsepower != '?']
df.horsepower = df.horsepower.astype(int)

### f ###
#Average of every column, except the name column
print("\n\nAverage of every column")
for col in df.columns:
    if col != 'name':
        print(col, df[col].mean().round(2))


################### Part 3 ###################
import matplotlib.pyplot as plt

### b ###
a = np.array([1,1,2,3,5,8,13,21,34])
b = np.array([1,8,28,56,70,56,28,8,1])

plt.plot(a, label='training accuracy')
plt.plot(b, label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


################### Part 4 ###################
import torch

a = torch.rand(3, 3)
b = torch.rand(3, 3)
print("\n\na: ", a)
print("b: ", b)
print(torch.matmul(a, b))