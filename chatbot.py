import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

#Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts the sentences into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, current_layer='Question1'):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    # Define allowed tags for each layer
    allowed_tags = {
        'Question1': ['good', 'bad', 'greetings'],
        'Question2a': ['Short-term', 'Long-term'],
        'Question2b': ['new', 'manageable', 'imediat_danger'],
        'Question3': ['mental_health','other_help'],
        'Question4': ['yes', 'no'],
        'Question5': ['anxiety', 'depression']
    }
    
    # Only process intents that are allowed in the current layer
    current_allowed_tags = allowed_tags.get(current_layer, [])
    
    for r in results:
        intent = classes[r[0]]
        # Only include intents that are allowed in the current layer
        if intent in current_allowed_tags:
            return_list.append({
                'intent': intent,
                'probability': str(r[1]),
                'layer': current_layer
            })
    
    return return_list
 
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that."
    
    # Get the highest probability intent
    current_intent = intents_list[0]
    tag = current_intent['intent']
    layer = current_intent['layer']
    
    # Process based on layer
    list_of_intents = intents_json['intents']
    response = None
    
    # Layer-specific processing
    if layer == 'Question1':
        # Basic interaction handling
        response = process_layer1(tag, list_of_intents)
    elif layer == 'Question2a':
        # Emotional assessment
        response = process_layer2(tag, list_of_intents)
    elif layer == 'Question2b':
        # Risk evaluation
        response = process_layer3(tag, list_of_intents)
    elif layer == 'Question3':
        # Support options
        response = process_layer4(tag, list_of_intents)
    elif layer == 'Question4':
        # Emergency handling
        response = process_layer5(tag, list_of_intents)
    elif layer == 'Question5':
        # Emergency handling
        response = process_layer5(tag, list_of_intents)
    
    return response if response else "I'm not sure how to respond to that."

# Layer-specific processing functions
def process_layer1(tag, list_of_intents):
    # Basic interaction processing
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

def process_layer2(tag, list_of_intents):
    # Emotional assessment processing
    for i in list_of_intents:
        if i['tag'] == tag:
            # You could add additional logic here to track emotional state
            return random.choice(i['responses'])
    return None

def process_layer3(tag, list_of_intents):
    # Risk evaluation processing
    for i in list_of_intents:
        if i['tag'] == tag:
            # Add severity checking logic here
            return random.choice(i['responses'])
    return None

def process_layer4(tag, list_of_intents):
    # Support options processing
    for i in list_of_intents:
        if i['tag'] == tag:
            # Add support recommendation logic here
            return random.choice(i['responses'])
    return None

def process_layer5(tag, list_of_intents):
    # Emergency handling processing
    for i in list_of_intents:
        if i['tag'] == tag:
            # Add emergency response logic here
            return random.choice(i['responses'])
    return None

print("COM727 Chatbot is here!")
print("How are you feeling?")
current_layer = 'Question1'

while True:
    message = input("You: ")
    ints = predict_class(message, current_layer)
    res = get_response(ints, intents)
    print(res)
    
    # Update layer based on response and specific intent
    if current_layer == 'Question1' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")  # Debug print
        
        # Route to different questions based on intent
        if detected_intent == 'bad':
            current_layer = 'Question2a'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
        elif detected_intent == 'good':
            current_layer = 'Question5'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
        else:
            current_layer = 'Question1'  # Route to regular follow-up
            print(f"Debug - Current layer: {current_layer}")  # Debug print


    elif current_layer == 'Question2a' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")  # Debug print
        
        # Route to different questions based on intent
        if detected_intent == 'Long-term':
            current_layer = 'Question3'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
        elif detected_intent == 'Short-term':
            current_layer = 'Question2b'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
        else:
            current_layer = 'Question2a'  # Route to regular follow-up
            print(f"Debug - Current layer: {current_layer}")  # Debug print
    
    elif current_layer == 'Question2b' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")  # Debug print
        
        # Route to different questions based on intent
        if detected_intent == 'new':
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        elif detected_intent == 'manageable':
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        elif detected_intent == 'imediat_danger':
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        else:
            current_layer = 'Question2b'  # Route to regular follow-up
            print(f"Debug - Current layer: {current_layer}")  # Debug print

    elif current_layer == 'Question3' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")  # Debug print
        
        # Route to different questions based on intent
        if detected_intent == 'mental_health':
            current_layer = 'Question4'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
        elif detected_intent == 'other_help':
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        else:
            current_layer = 'Question3'  # Route to regular follow-up
            print(f"Debug - Current layer: {current_layer}")  # Debug print

    elif current_layer == 'Question4' and ints:
        # Get the detected intent
        detected_intent = ints[0]['intent']
        print(f"Debug - Detected intent: {detected_intent}")  # Debug print
        
        # Route to different questions based on intent
        if detected_intent == 'anxiety':
            current_layer = 'Question4'  # Route to immediate danger assessment
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        elif detected_intent == 'depression':
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            break
        else:
            current_layer = 'Question4'  # Route to regular follow-up
            print(f"Debug - Current layer: {current_layer}")  # Debug print
            print("Due to limited knowlelodge we could not help you identify the problem but we would recommend to stay in continued contact with your gp")