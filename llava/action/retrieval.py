import os
import numpy as np
from openai import OpenAI
from typing import List, Tuple
import json
from collections import defaultdict
import os

class SemanticRetriever:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the retriever with OpenAI API key and model.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.model = model
        self.embeddings = []
        self.sentences = []
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Get embedding for a single piece of text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def add_sentences(self, sentences: List[str]) -> None:
        """
        Add multiple sentences to the retrieval system.
        
        Args:
            sentences (List[str]): List of sentences to add
        """
        for sentence in sentences:
            embedding = self.embed_text(sentence)
            self.embeddings.append(embedding)
            self.sentences.append(sentence)
            
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a (np.ndarray): First vector
            b (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def find_similar(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar sentences to the query.
        
        Args:
            query (str): Query sentence
            top_k (int): Number of similar sentences to return
            
        Returns:
            List[Tuple[str, float]]: List of (sentence, similarity_score) pairs
        """
        query_embedding = self.embed_text(query)
        
        # Calculate similarities
        similarities = [
            (sentence, self.cosine_similarity(query_embedding, stored_embedding))
            for sentence, stored_embedding in zip(self.sentences, self.embeddings)
        ]
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the current state to a file.
        
        Args:
            filepath (str): Path to save the state
        """
        state = {
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'sentences': self.sentences
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load state from a file.
        
        Args:
            filepath (str): Path to load the state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.embeddings = [np.array(emb) for emb in state['embeddings']]
        self.sentences = state['sentences']

# Example usage:
if __name__ == "__main__":
    # Initialize retriever

    retriever = SemanticRetriever()
    
    # Add some example sentences
    example_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The weather is beautiful today.",
        "OpenAI develops advanced language models.",
        "Deep learning has revolutionized computer vision.",
        "Natural language processing helps computers understand text.",
        "The sun rises in the east.",
        "Artificial intelligence is changing the world.",
        "Data science combines statistics and programming."
    ]
    
    retriever.add_sentences(example_sentences)
    
    # Find similar sentences
    query = "AI and machine learning are transforming technology"
    similar_sentences = retriever.find_similar(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    print("Most similar sentences:")
    for sentence, score in similar_sentences:
        print(f"Score: {score:.4f} - {sentence}")
    
    # Save state
    retriever.save_to_file("retriever_state.json")