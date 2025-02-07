import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

def clean_text(text):
    """Clean and preprocess text data."""
    # Prepare resources
    stopwords_list = stopwords.words('english')
    #lemmatizer = WordNetLemmatizer()

    filter_words = ['amazon', 'apex', 'assassin', 'battlefield', 'bfbd', 'bioshock', 'borderland', 
                    'borderlands', 'call', 'com', 'creed', 'csgo', 'dead', 'doom', 'duty', 'facebook', 
                    'fifa', 'fortnite', 'game', 'google', 'gta', 'halo', 'hearts', 'instagram', 
                    'johnson', 'kingdom', 'league', 'legends', 'mario', 'microsoft', 'minecraft', 
                    'nintendo', 'nividia', 'overwatch', 'pic', 'pokemon', 'pubg', 'red', 'reddit', 
                    'redemption', 'rocket', 'super', 'tv', 'twitch', 'twitter', 'verizon', 'xbox', 'youtube']

    # Preprocessing steps
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove digits

    # Convert to lowercase and tokenize
    text = text.lower()
    words_in_text = word_tokenize(text)

    # Apply POS tagging and filter tokens
    tagged_tokens = pos_tag(words_in_text)
    filtered_words = [
        #lemmatizer.lemmatize(word, get_wordnet_pos(tag)) # Lemmatize words
        word for word, tag in tagged_tokens
        #if get_wordnet_pos(tag) in [wordnet.ADJ, wordnet.VERB, wordnet.ADV] # Filter only adjectives, verbs, and adverbs
        if word not in filter_words
        and len(word) > 2
        and word not in stopwords_list
        and not word.startswith('_')
    ]

    return ' '.join(filtered_words)
