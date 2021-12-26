from chat_service import _ChatBot

myChatBot = _ChatBot()
print("Lets Chat! type quit to exit")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    response = myChatBot.getResponse(sentence)
    print(f'Yippers: {response}')

