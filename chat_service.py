import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize


# inputSize = data["input_size"]
#     hiddenSize = data["hidden_size"]
#     outputSize = data["output_size"]
#     all_words = data["all_words"]
#     tags = data["tags"]
#     modelState = data["model_state"]

def loadIntents():
    with open('intents.json','r') as f:
        intents = json.load(f)
    return intents

def loadData(File = "data.pth"):
    data = torch.load(File, map_location='cpu')
    return data
def modelInit(data,device):
    model = NeuralNet(data["input_size"],data['hidden_size'],data['output_size']).to(device)
    modelState = data['model_state']
    model.load_state_dict(modelState)
    model.eval()
    return model

class _ChatBot(NeuralNet):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.device = torch.device('cpu')
        self.intents = loadIntents()
        self.data = loadData()
        self.all_words = self.data['all_words']
        self.tags = self.data['tags']
        self.model = modelInit(self.data,self.device)
    
    def getResponse(self,message):
        tokenized_sentence = tokenize(message)
        bow = bag_of_words(tokenized_sentence,self.all_words)
        bow = bow.reshape(1,bow.shape[0]) # flatten
        bow = torch.from_numpy(bow).to(self.device)

        output = self.model(bow)
        _,predicted = torch.max(output,dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output,dim=1)
        actualProb = probs[0][predicted.item()]

        if actualProb > .75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    botresponse = random.choice(intent['responses'])
                    return botresponse
        else:
            return "I Am Sorry I Do Not Understand"

myChatBot = _ChatBot()