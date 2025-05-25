import re
import string
from collections import Counter
import pandas as pd
import fasttext

# Load FastText language identification model
fasttext_model = fasttext.load_model("model/lid.176.ftz")  # Adjust path if needed

# Load review CSV
input_path = "data/processed/multi/filtered/all.csv"
df = pd.read_csv(input_path)
df = df[df['language']=='english']

# Ensure the review column exists
if 'review' not in df.columns:
    raise ValueError("The CSV must contain a 'review' column.")

# Common Nepali words in Romanized form
NEPALI_WORDS = [
    'ma', 'timro', 'hami', 'yo', 'tyo', 'ke', 'kasto', 'kasari', 'cha', 'chan', 
    'chhu', 'ho', 'hoina', 'ra', 'pani', 'lai', 'bata', 'ko', 'ka', 'le', 'mero',
    'tapai', 'hamro', 'khana', 'paani', 'ghar', 'baato', 'didi', 'bhai',
    'aama', 'buwa', 'hijo', 'aaja', 'bholi', 'ramro', 'naramro', 'sano', 'thulo',
    'huncha', 'bhayeko', 'garne', 'jasto', 'yesto', 'kina', 'bhane', 'tara', 'ani',
    'bhanda', 'haina', 'thiyo', 'aaunu', 'jaanu', 'hernu', 'sunnu', 'bujhnu', 'bhannu',
    'xa', 'xaina', 'hunxa', 'payeko', 'lagyo', 'aadha', 'pura', 'sabai', 'dherai', 'ali',
    'kam', 'garna', 'milxa', 'milchha', 'paxi', 'nai', 'hudo', 'raixa', 'rahexa', 'din',
    'aaghadi', 'pachhadi', 'paxadi', 'bichma', 'bichar', 'vayena', 'vayo', 'babal', 'dami',
    'ekdam', 'aayana', 'magayeko', 'thiye', 'aayexa', 'aayo', 'nalaagni', 'nalaagne', 'lagne',
    'soche', 'jastai', 'lagcha', 'barema', 'bhanu', 'jati', 'ni', 'thorai', 'dekheyna', 'dekhyo',
    'saman', 'chahe', 'pathaidera', 'maal', 'sajilo', 'halna', 'vaya', 'kinera', 'hawa', 'falxa',
    'khasai', 'rakhyo', 'chalxa', 'majjale', 'maila', 'garyako', 'hatma', 'theyo', 'hunuhudo',
    'pahela', 'yati', 'bajya', 'aaudaixu', 'vanera', 'dinu', 'huni', 'milayera', 'feri', 'hajur', 
    'haruko', 'kinnexu', 'liyera', 'hola', 'xito', 'pathaidinu', 'vitrai', 'kucheko', 
    'sastra', 'bhagavad', 'gita', 'gyan', 'dhyan', 'bhakti', 'afai', 'garda', 'padna', 
    'dammi', 'kaam', 'garxa', 'fone', 'maa', 'ramrari', 'yadi', 'chalena', 'vane', 'gayera',
    'garnu', 'chaliraxa', 'chalirahekoxa', 'ramrai', 'garirako', 'garirakchu', 'jhos', 'ahile',
    'vayesi', 'milne', 'yo', 'eauta', 'matra', 'pathaunu', 'vayecha', 'khai', 'ekdamai', 'majale', 
    'malai', 'maan', 'pareko','kura', 'vaneko', 'mildo', 'kunai', 'jodna', 'pardaina', 'garera', 'herna', 
    'saknu', 'tesko', 'lagi', 'ya', 'vanau', 'chahinxa' 'jun', 'sajilai', 'kinna', 'yar', 'khojnu', 'bho', 
    'mani', 'pauxa', 'tesle', 'tapaile', 'bas', 'eutamatra', 'kami', 'gardaina', 'anusar', 'ek', 'dui', 'jana', 
    'chai', 'hune', 'tarkari', 'pakauna', 'lehengako', 'linu', 'parda', 'dekhincha', 'chahi', 'gare', 'sanga', 
    'apnai', 'choicema', 'banauna', 'milcha', 'mil६', 'halnu', 'parcha', 'nabhaye', 'khasne', 'dekhako', 'dinubhayena',
    'paila', 'garcha', 'bhayo', 'raicha', 'paryo', 'aucha', 'lekheko', 'ayena', 'mahile', 'gareko', 'vayera',
    'pathaayo', 'mahele', 'firta', 'pathai', 'deko', 'bigareko', 'gardina', 'thyeee', 'bigreko', 'thyo'
]

# Common word variants
NEPALI_STANDARD_SPELLING = {
        'ramroo': 'ramro', 'rmro': 'ramro', 'raamro': 'ramro',
        'dammi': 'dami', 'dammee': 'dami', 'daami': 'dami', 'daamee': 'dami', 'damehh': 'dami',
        'ekdum': 'ekdam', 'ekdamm': 'ekdam', 'yakdam': 'ekdam',
        'ekdumai': 'ekdamai',
        'thikaai': 'thikai', 'thikey': 'thikai',
        'dherei': 'dherai', 'dherey': 'dherai', 'derai': 'dherai', 'derey': 'dherai',
        'lageo': 'lagyo', 'laagyo': 'lagyo', 'lago': 'lagyo',
        'hunchha': 'huncha', 'hunxa': 'huncha',
        'raichha': 'raicha', 'raixa': 'raicha', 'raix': 'raicha', 'rahex': 'raicha',
        'sakinchha': 'sakincha', 'sakinxa': 'sakincha',
        'anusaar': 'anusar', 'aanusar': 'anusar',
        'vayo': 'bhayo', 'bhaeo': 'bhayo',
        'vae': 'bhae',
        'haroo': 'haru',
        'anee': 'ani',
        'taraa': 'tara',
        'panee': 'pani',
        'tapain': 'tapai', 'tapaai': 'tapai',
        'malaai': 'malai',
        'mailee': 'maile', 'mael': 'maile',
        'auxa': 'aucha', 'aaucha': 'aucha', 'aauxa': 'aucha',
        # Common phrases
        'ke xa': 'ke cha',
        'k xa': 'k cha',
        'kasto xa': 'kasto cha',
        'ksto xa': 'kasto cha',
        'thik xa': 'thik cha',
        'thikai xa': 'thikai cha',
        'ramro xa': 'ramro cha',
        'ramro xaina': 'ramro chaina',
        'xa': 'cha',
        'xaina': 'chaina',
        'timlaai': 'timilai', 'timila': 'timilai',
        'sanchho': 'sancho', 'sanxo': 'sancho',
        'vayena': 'bhayena', 'bhayaena': 'bhayena',
        'aaja': 'aja',
        'offis': 'office', 'offs': 'office', 'ophice': 'office',
        'vayenaki': 'bhayenaki'
    }

# Common Nepali suffixes in Romanized form
NEPALI_SUFFIXES = [
    'ko', 'le', 'lai', 'ma', 'bata', 'sanga', 'haru', 'chha', 'nai', 'pani',
    'chhau', 'chhu', 'bhayo', 'bhayeko', 'huncha', 'thyo', 'eko', 'ne', 'dai',
    'xa', 'xaina', 'hunxa', 'hunthyo'
]

# Common Nepali conjunctions and postpositions
NEPALI_CONJUNCTIONS = [
    'ra', 'ani', 'tara', 'ki', 'baru', 'yedi', 'bhane', 'kinaki', 'kinabhane',
    'tesaile', 'tyasaile', 'tespachi', 'yesogarera'
]

# Specific Nepali character combinations that rarely appear in English
NEPALI_CHAR_PATTERNS = [
    'aa', 'chh', 'kh', 'th', 'dh', 'bh', 'ph', 'gh', 'jh'
]

def is_romanized_nepali(text):
    """
    Determines if a given text is likely to be Romanized Nepali.
    Returns a probability score (0-1) and the features that contributed to the decision.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = text.split()
    
    if len(words) == 0:
        return 0, []
    
    # Calculate features
    features = []
    
    # Feature 1: Percentage of words that match common Nepali words
    nepali_word_matches = sum(1 for word in words if word in NEPALI_WORDS)
    nepali_word_ratio = nepali_word_matches / len(words)
    if nepali_word_ratio > 0:
        features.append(f"Found {nepali_word_matches} common Nepali words")
    
    # Feature 2: Percentage of words that end with Nepali suffixes
    suffix_matches = sum(1 for word in words if any(word.endswith(suffix) for suffix in NEPALI_SUFFIXES))
    suffix_ratio = suffix_matches / len(words)
    #if suffix_ratio > 0:
    #    features.append(f"Found {suffix_matches} words with Nepali suffixes")
    
    # Feature 3: Presence of Nepali conjunctions
    conjunction_matches = sum(1 for word in words if word in NEPALI_CONJUNCTIONS)
    conjunction_ratio = conjunction_matches / len(words)
    #if conjunction_ratio > 0:
    #    features.append(f"Found {conjunction_matches} Nepali conjunctions")
    
    # Feature 4: Character patterns typical in Romanized Nepali
    char_pattern_count = sum(1 for pattern in NEPALI_CHAR_PATTERNS if pattern in text)
    char_pattern_density = char_pattern_count / (len(text) + 0.1)  # Avoid division by zero
    #if char_pattern_count > 0:
    #    features.append(f"Found {char_pattern_count} Nepali character patterns")
    
    # Calculate the overall score
    # Weighted sum of features (can be adjusted based on importance)
    score = (
        0.4 * nepali_word_ratio + 
        0.3 * suffix_ratio + 
        0.2 * conjunction_ratio + 
        0.1 * char_pattern_density * 10  # Multiplier to bring it to a similar scale
    )

    # Classify as Nepali if score is above threshold
    return score > 0.25  # This threshold can be adjusted


# Language classification function
def classify_language(text):
    prediction = fasttext_model.predict(text.strip().replace('\n', ' '), k=1)
    label = prediction[0][0].replace("__label__", "")
    if label == "en":
        if(is_romanized_nepali(text)):
            return "romanized_nepali"
        else: 
            return "english"
    elif label == "ne":
        return "nepali"
    else:
        return "romanized_nepali"

# Apply language classification
df['new_language'] = df['review'].astype(str).apply(classify_language)

# Save result to new CSV
output_path = "data/processed/multi/filtered/eng_lang_chk.csv"
df.to_csv(output_path, index=False)

print(f"Language detection completed. Output saved to: {output_path}")


