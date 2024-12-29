# Import required NLP libraries for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
# nltk.download('all')  # Uncomment to download all NLTK data
nltk.download('punkt')    # For tokenization
nltk.download('wordnet')  # For lemmatization
nltk.download('stopwords')  # For filtering common words
nltk.download('omw-1.4')   # Open Multilingual Wordnet

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Remove negation words from stopwords to preserve meaning
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
stop_words = stop_words - negation_words

def text_to_pattern(text, ignore_chars=None):
    """
    Transforms normal human text into a structured pattern.
    Args:
        text (str): Raw input text to be processed
        ignore_chars (list): Characters to remove from text (default: ['?', '!', '.', ','])
    
    Returns:
        list: Processed words forming the pattern, maintaining negations
    """
    # Set default characters to ignore if none provided
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
human_text = "I'm happy Feeling great I'm so glad Life is amazing So excited Feeling awesome Everything is perfect Wonderful Fantastic Amazing Superb Fabulous Excited I'm over the moon"
pattern = text_to_pattern(human_text)
print("Pattern:", pattern)