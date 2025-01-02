# Import required libraries for loading the model and uterlizing it
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize the lemmatizer and load the trained model and intents files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Tokenize and lemmatize the users input 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert input sentence to bag of words array 
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Pridicts the probability of a tag and filters it based on the question
def predict_class(sentence, current_layer='Question1'):
    # Pridicts the probability of a tag
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    # Define allowed tags for each question tp determain the flow of the programme
    allowed_tags = {
        'Question1': ['good', 'bad', 'greetings'],
        'Question2a': ['Short-term', 'Long-term'],
        'Question2b': ['new', 'manageable', 'imediat_danger'],
        'Question3': ['mental_health'],
        'Question4': ['yes', 'no'],
        'Question5': ['anxiety', 'depression']
    }
    
    # Filtering out the allowed tags
    current_allowed_tags = allowed_tags.get(current_layer, [])
    for r in results:
        intent = classes[r[0]]
        if intent in current_allowed_tags:
            return_list.append({
                'intent': intent,
                'probability': str(r[1]),
                'layer': current_layer
            })
    
    return return_list
 
# Gets the out put of the model
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that."
    
    # Get the intent that fit the best
    current_intent = intents_list[0]
    tag = current_intent['intent']
    layer = current_intent['layer']
    list_of_intents = intents_json['intents']
    response = None
    
    # sorts based on the previous layer
    if layer == 'Question1':
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question2a':
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question2b':
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question3':
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question4':
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question5':
        response = process_layer1(tag, list_of_intents)
    return response if response else "I'm not sure how to respond to that."

# Processes tjhe 
def process_layer1(tag, list_of_intents):
    # Process initial user input and determine emotional state
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

# Start the chatbot and first question
print("COM727 Chatbot is here!")
print("How are you feeling?")
current_layer = 'Question1'

while True:
    # Asks for the users input
    message = input("You: ")
    ints = predict_class(message, current_layer)
    res = get_response(ints, intents)
    print(res)
    
    # Update layer based on response and specific intent
    if current_layer == 'Question1' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")
        
        # Route to different questions based on intent
        if detected_intent == 'bad':
            current_layer = 'Question2a' 
            print(f"Debug - Current layer: {current_layer}") 
        elif detected_intent == 'good':
            current_layer = 'Question5'
            print(f"Debug - Current layer: {current_layer}") 
        else:
            current_layer = 'Question1'
            print(f"Debug - Current layer: {current_layer}")


    elif current_layer == 'Question2a' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}") 
        
        # Route to different questions based on intent
        if detected_intent == 'Long-term':
            current_layer = 'Question3'  
            print(f"Debug - Current layer: {current_layer}")  
        elif detected_intent == 'Short-term':
            current_layer = 'Question2b'  
            print(f"Debug - Current layer: {current_layer}")  
        else:
            current_layer = 'Question2a'  
            print(f"Debug - Current layer: {current_layer}")  
    
    elif current_layer == 'Question2b' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}") 
        
        # Route to different questions based on intent
        if detected_intent == 'new':
            print(f"Debug - Current layer: {current_layer}") 
            break
        elif detected_intent == 'manageable':
            print(f"Debug - Current layer: {current_layer}")
            break
        elif detected_intent == 'imediat_danger':
            print(f"Debug - Current layer: {current_layer}") 
            break
        else:
            current_layer = 'Question2b'
            print(f"Debug - Current layer: {current_layer}") 

    elif current_layer == 'Question3' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}") 
        
        # Route to different questions based on intent
        if detected_intent == 'mental_health':
            current_layer = 'Question4'  
            print(f"Debug - Current layer: {current_layer}") 
        else:
            current_layer = 'Question3' 
            print(f"Debug - Current layer: {current_layer}") 
            print("Contact your local crisis team/service as they are great at tackerling the larger problems")

    elif current_layer == 'Question4' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}") 
        
        # Route to different questions based on intent
        if detected_intent == 'yes':
            current_layer = 'Question5'  
            print(f"Debug - Current layer: {current_layer}") 
        elif detected_intent == 'no':
            print(f"Debug - Current layer: {current_layer}")  
            break
        else:
            current_layer = 'Question4' 
            print(f"Debug - Current layer: {current_layer}")  

    elif current_layer == 'Question5' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}") 
        
        # Route to different questions based on intent
        if detected_intent == 'anxiety':
            current_layer = 'Question5'  
            print(f"Debug - Current layer: {current_layer}") 
            break
        elif detected_intent == 'depression':
            current_layer = 'Question5'  
            print(f"Debug - Current layer: {current_layer}")  
            break
        else:
            current_layer = 'Question5' 
            print(f"Debug - Current layer: {current_layer}")  
            print("Due to limited knowlelodge we could not help you identify the problem but we would recommend to stay in continued contact with your gp")
            break