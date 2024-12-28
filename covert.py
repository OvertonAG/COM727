import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
#nltk.download('all')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Remove negation words from stopwords
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
stop_words = stop_words - negation_words

def text_to_pattern(text, ignore_chars=None):
    """
    Transforms normal human text into a structured pattern.
    Args:
        text (str): Input human-readable text.
        ignore_chars (list): List of characters to remove.
    
    Returns:
        list: List of processed words forming the pattern.
    """
    if ignore_chars is None:
        ignore_chars = ['?', '!', '.', ',']

    # Lowercase and remove special characters
    text = text.lower()
    for char in ignore_chars:
        text = text.replace(char, '')

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words and lemmatize
    processed_words = [lemmatizer.lemmatize(word, pos='v') for word in words if word not in stop_words]
    return processed_words

# Example Usage
human_text = "I'm happy Feeling great I'm so glad Life is amazing So excited Feeling awesome Everything is perfect Wonderful Fantastic Amazing Superb Fabulous Excited I'm over the moon"
pattern = text_to_pattern(human_text)
print("Pattern:", pattern)