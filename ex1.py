import spacy
from datasets import load_dataset
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple
import numpy as np

class LanguageModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.total_words = 0
        self.vocab = set()
        
    def preprocess_text(self, text: str) -> List[str]:
        """Convert text to list of lemmatized words, filtering out non-alpha tokens."""
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if token.is_alpha]
    
    def train(self, dataset):
        """Train both unigram and bigram models on the dataset."""
        for item in dataset:
            if not item['text'].strip():  # Skip empty lines
                continue
                
            # Process each line as a separate document
            tokens = self.preprocess_text(item['text'])
            if not tokens:  # Skip if no valid tokens
                continue
                
            # Add START token for bigram model
            tokens = ['<START>'] + tokens
            
            # Update counts
            self.total_words += len(tokens) - 1  # Don't count START
            
            # Unigram counts (excluding START token)
            self.unigram_counts.update(tokens[1:])
            
            # Bigram counts
            for i in range(len(tokens)-1):
                self.bigram_counts[tokens[i]][tokens[i+1]] += 1
            
            # Update vocabulary
            self.vocab.update(tokens[1:])  # Don't include START in vocab
    
    def unigram_probability(self, word: str) -> float:
        """Calculate unigram probability P(word)."""
        return math.log(self.unigram_counts[word] / self.total_words) if word in self.unigram_counts else float('-inf')
    
    def bigram_probability(self, word1: str, word2: str) -> float:
        """Calculate bigram probability P(word2|word1)."""
        if word1 not in self.bigram_counts or word2 not in self.bigram_counts[word1]:
            return float('-inf')
        return math.log(self.bigram_counts[word1][word2] / sum(self.bigram_counts[word1].values()))
    
    def interpolated_probability(self, word1: str, word2: str, lambda_bigram: float = 2/3) -> float:
        """Calculate interpolated probability between bigram and unigram models."""
        bigram_prob = math.exp(self.bigram_probability(word1, word2)) if word1 in self.bigram_counts else 0
        unigram_prob = math.exp(self.unigram_probability(word2)) if word2 in self.unigram_counts else 0
        
        return math.log(lambda_bigram * bigram_prob + (1 - lambda_bigram) * unigram_prob) if (bigram_prob or unigram_prob) else float('-inf')
    
    def predict_next_word(self, context: str) -> str:
        """Predict the most probable next word given a context."""
        tokens = self.preprocess_text(context)
        if not tokens:
            return None
        
        last_word = tokens[-1]
        if last_word not in self.bigram_counts:
            return None
        
        return max(self.bigram_counts[last_word].items(), key=lambda x: x[1])[0]
    
    def sentence_probability(self, sentence: str, model_type: str = 'bigram') -> float:
        """Calculate the log probability of a sentence under the specified model."""
        tokens = self.preprocess_text(sentence)
        if not tokens:
            return float('-inf')
        
        log_prob = 0
        
        if model_type == 'unigram':
            for token in tokens:
                log_prob += self.unigram_probability(token)
                
        elif model_type == 'bigram':
            tokens = ['<START>'] + tokens
            for i in range(len(tokens)-1):
                log_prob += self.bigram_probability(tokens[i], tokens[i+1])
                
        else:  # interpolated
            tokens = ['<START>'] + tokens
            for i in range(len(tokens)-1):
                log_prob += self.interpolated_probability(tokens[i], tokens[i+1])
        
        return log_prob
    
    def perplexity(self, sentences: List[str], model_type: str = 'bigram') -> float:
        """Calculate perplexity over a set of sentences."""
        total_log_prob = 0
        total_words = 0
        
        for sentence in sentences:
            tokens = self.preprocess_text(sentence)
            if not tokens:
                continue
            
            total_words += len(tokens)
            total_log_prob += self.sentence_probability(sentence, model_type)
            
        if total_words == 0:
            return float('inf')
            
        return math.exp(-total_log_prob / total_words)

# Main execution
def main():
    # Load dataset
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    
    # Initialize and train model
    lm = LanguageModel()
    lm.train(text)
    
    # Task 2: Continue the sentence
    context = "I have a house in"
    next_word = lm.predict_next_word(context)
    print(f"\nTask 2 - Most probable word after '{context}': {next_word}")
    
    # Task 3: Compute probabilities and perplexity
    sentences = [
        "Brad Pitt was born in Oklahoma",
        "The actor was born in USA"
    ]
    
    print("\nTask 3 - Bigram model:")
    for sentence in sentences:
        prob = lm.sentence_probability(sentence, 'bigram')
        print(f"Log probability of '{sentence}': {prob:.4f}")
    
    perplexity = lm.perplexity(sentences, 'bigram')
    print(f"Perplexity of both sentences: {perplexity:.4f}")
    
    # Task 4: Interpolated model
    print("\nTask 4 - Interpolated model:")
    for sentence in sentences:
        prob = lm.sentence_probability(sentence, 'interpolated')
        print(f"Log probability of '{sentence}': {prob:.4f}")
    
    perplexity = lm.perplexity(sentences, 'interpolated')
    print(f"Perplexity of both sentences: {perplexity:.4f}")

if __name__ == "__main__":
    main()