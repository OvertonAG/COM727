# Import required libraries for data processing, NLP, and deep learning
import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK data for text processing
nltk.download('punkt')      # For tokenization
nltk.download('wordnet')    # For lemmatization
nltk.download('stopwords')  # For removing common words

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load conversation patterns and responses from JSON file
intents = json.loads(open('intents.json').read())

# Initialize lists to store processed words and intent classes
words = []          # All unique words from patterns
classes = []        # All unique intent tags
documents = []      # Combinations of patterns and their intents
ignore_letters = ['?', '!', '.', '/', '@']  # Characters to filter out

# Process each pattern in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern into individual words
        word_list = nltk.word_tokenize(pattern)
        # Add words to the words list
        words.extend(word_list)
        # Add pattern and its associated tag to documents
        documents.append((word_list, intent['tag']))
        # Add tag to classes list if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Get English stop words for filtering
stop_words = set(stopwords.words('english'))

# Process words: lemmatize and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Remove duplicates and sort
words = sorted(set(words))

# Sort classes for consistency
classes = sorted(set(classes))

# Save processed words and classes to files for the chatbot to use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)  # Create empty output array for each class

# Create training data with bag of words for each pattern
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatize and lowercase each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words array
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create output row with '1' for current tag and '0' for other tags
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split training data into input (X) and output (Y) variables
train_x = list(training[:, 0])  # Patterns
train_y = list(training[:, 1])  # Tags

# Create neural network model
model = Sequential()
# Input layer with 128 neurons
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
# Hidden layer with 64 neurons
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output layer with neurons equal to number of intents
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configure model training parameters
# Using Adam optimizer instead of SGD for better performance
model.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), 
                epochs=100, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.keras', hist)
