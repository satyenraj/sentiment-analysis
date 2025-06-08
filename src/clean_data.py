import re
import pandas as pd

class TextCleaner:
    def __init__(self):
        # Define regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F700-\U0001F77F"  # alchemical symbols
                                    u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                    u"\U00002702-\U000027B0"  # Dingbats
                                    u"\U000024C2-\U0001F251" 
                                    "]+")


    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning applicable to all languages.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Replace emojis with space (keep them separated)
        text = self.emoji_pattern.sub(' ', text)
        
        # Replace newlines and carriage returns with space
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    input_file = 'data/review_nepali-001.csv' 
    df = pd.read_csv(input_file)

    # Step 2: Clean review text
    cleaner = TextCleaner()
    df['review'] = df['review'].apply(cleaner.clean_text)

    # Step 3: Save to new CSV
    output_file = 'data/cleaned_review_nepali-001.csv'
    df.to_csv(output_file, index=False)

    print(f"Cleaned reviews saved to {output_file}")

