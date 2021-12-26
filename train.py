import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataset
from model import NeuralNet
from tqdm import tqdm
with open('intents.json','r') as file:
    intents = json.load(file)


all_words = []
tags = []
pattern_and_tags = []
intents = intents['intents']
for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)
        pattern_and_tags.append((words,tag))

ignore = [',', '!', '.', '?']
all_words = [stem(word) for word in all_words if word not in ignore]

all_words = sorted(set(all_words))
tags = sorted(tags)

xtrain = []
ytrain = []
for pattern,tag in pattern_and_tags:
    bag = bag_of_words(pattern,all_words)
    xtrain.append(bag)

    label = tags.index(tag)
    ytrain.append(label)



xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

class ChatDataSet(Dataset):
    def __init__(self) -> None:
        self.samples = len(xtrain)
        self.xdata = xtrain
        self.ydata = ytrain
    
    def __getitem__(self, index):
        return self.xdata[index], self.ydata[index]
    
    def __len__(self):
        return self.samples

batchSize = 8

hiddenSize = 8
inputSize = len(xtrain[0]) # or xtrain[0]
outputSize = len(tags)
learningRate = 0.001

numEpocs = 400
dataset = ChatDataSet()
train_load = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)

device = torch.device('cpu')
#print('Using gpu' if torch.cuda.is_available() else 'Using cpu')

model = NeuralNet(inputSize,hiddenSize,outputSize).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in tqdm(range(numEpocs)):
    for (words,labels) in train_load:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)

        #forward
        outputs = model(words)
        loss = criterion(outputs,labels)

        #backward and optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        #print(f'\n epoch {epoch + 1}/{numEpocs}, loss = {loss.item():.4f}')
        pass

#print(f'final loss = {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": inputSize,
    "output_size": outputSize,
    "hidden_size": hiddenSize,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)

#print('Training Complete')
