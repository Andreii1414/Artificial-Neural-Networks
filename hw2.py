import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import requests
import pickle, gzip, numpy as np

url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

#Concatenare train si test
train_x, train_y = train_set
train_x = np.concatenate([train_x, test_set[0]])
train_y = np.concatenate([train_y, test_set[1]])

#Set de validare
valid_x, valid_y = valid_set

#Transformare in tensori
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
valid_x = torch.tensor(valid_x, dtype=torch.float32)
valid_y = torch.tensor(valid_y, dtype=torch.long)

#Parametrii
input_size = 28 * 28
hidden_size = 128
output_size = 10
lr = 0.001
batch_size = 64
epochs = 10

#Modelul
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.Softmax(dim=1)
    )

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_x), batch_size):
        end = i + batch_size
        inputs, labels = train_x[i:end], train_y[i:end]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()

train_outputs = model(train_x)
train_preds = torch.argmax(train_outputs, dim=1).numpy()
train_acc = accuracy_score(train_y.numpy(), train_preds)
train_f1 = f1_score(train_y.numpy(), train_preds, average="weighted")

valid_outputs = model(valid_x)
valid_preds = torch.argmax(valid_outputs, dim=1).numpy()
valid_acc = accuracy_score(valid_y.numpy(), valid_preds)
valid_f1 = f1_score(valid_y.numpy(), valid_preds, average="weighted")

print("Train accuracy: ", train_acc)
print("Train f1: ", train_f1)
print("Valid accuracy: ", valid_acc)
print("Valid f1: ", valid_f1)