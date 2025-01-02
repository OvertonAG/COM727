# Import required libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('all') 
nltk.download('punkt')    
nltk.download('wordnet')  
nltk.download('stopwords') 
nltk.download('omw-1.4')   

# Initialize NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Remove negation words from stopwords
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
stop_words = stop_words - negation_words

def text_to_pattern(text, ignore_chars=None):
    # Set default characters to ignore
    if ignore_chars is None:
        ignore_chars = ['?', '!', '.', ',']

    # Convert to lowercase and remove specified characters
    text = text.lower()
    for char in ignore_chars:
        text = text.replace(char, '')

    # Split text into individual words
    words = word_tokenize(text)

    # Process words: remove stopwords (except negations) and lemmatize
    processed_words = [lemmatizer.lemmatize(word, pos='v') 
                      for word in words if word not in stop_words]
    return processed_words

# Example Usage with emotional text
human_text = "I'm happy feeling great"
pattern = text_to_pattern(human_text)
print("Pattern:", pattern)