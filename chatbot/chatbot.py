import nltk
from nltk.chat.util import Chat, reflections

pairs = [
    [r"hi|hello|hey", ["Hello! How can I help you today?", "Hi there!"]],
    [r"what is your name?", ["I'm a chatbot created with Python."]],
    [r"how are you?", ["I'm doing great, thanks!", "All good!"]],
    [r"bye", ["Goodbye!", "See you later!"]],
]

chatbot = Chat(pairs, reflections)
print("Chatbot ready! Type 'bye' to exit.")
chatbot.converse()