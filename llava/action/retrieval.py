import os
import numpy as np
import openai
from openai import OpenAI
from typing import List, Tuple
import json
from collections import defaultdict
import os
import csv
import pandas as pd

from typing import List, Tuple
import numpy as np
import json
from openai import OpenAI
import os
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor

class SemanticRetriever:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        """
        Initialize the retriever with OpenAI API key and model.
        
        Args:
            model (str): OpenAI embedding model to use
            batch_size (int): Number of texts to embed in parallel
        """
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.model = model
        self.batch_size = batch_size
        self.official_key_embeddings = None  # Will be numpy array
        self.gt_narration_embeddings = None  # Will be numpy array
        self.gt_narrations = []
        self.official_keys = []
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of texts in parallel.
        
        Args:
            texts (List[str]): Texts to embed
            
        Returns:
            np.ndarray: Matrix of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([data.embedding for data in response.data])
    
    def get_embeds(self, official_key_list: List[str], gt_narration_list: List[str]) -> None:
        """
        Add multiple sentences to the retrieval system using batched processing.
        
        Args:
            official_key_list (List[str]): List of official keys
            gt_narration_list (List[str]): List of ground truth narrations
        """
        self.official_keys = official_key_list
        self.gt_narrations = gt_narration_list
        
        # Process official keys in batches
        official_embeddings = []
        for i in tqdm(range(0, len(official_key_list), self.batch_size)):
            batch = official_key_list[i:i + self.batch_size]
            batch_embeddings = self.embed_batch(batch)
            official_embeddings.append(batch_embeddings)
        
        # Process narrations in batches
        narration_embeddings = []
        for i in tqdm(range(0, len(gt_narration_list), self.batch_size)):
            batch = gt_narration_list[i:i + self.batch_size]
            batch_embeddings = self.embed_batch(batch)
            narration_embeddings.append(batch_embeddings)
            
        # Convert to numpy arrays
        self.official_key_embeddings = np.vstack(official_embeddings)
        self.gt_narration_embeddings = np.vstack(narration_embeddings)

    def find_similar(self, queries: List[str], top_k: int = 10, search_type: str = 'both', 
                        batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """
        Find the top_k most similar items for multiple queries using batched operations.
        
        Args:
            queries (List[str]): List of query texts
            top_k (int): Number of similar items to return per query
            search_type (str): 'official', 'narration', or 'both'
            batch_size (int): Size of batches for API calls
            
        Returns:
            List[List[Tuple[str, float]]]: List of results for each query, where each result is 
                                        a list of (text, similarity_score) pairs
        """
        # Process queries in batches to get embeddings
        all_query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i + batch_size]
            batch_embeddings = self.embed_batch(batch)
            all_query_embeddings.extend(batch_embeddings)
        
        query_embeddings = np.array(all_query_embeddings)
        results = []
        
        # Compute similarities for all queries at once using matrix multiplication
        if search_type in ['official', 'both']:
            # Shape: (num_queries, num_embeddings)
            similarities = np.dot(query_embeddings, self.official_key_embeddings.T) / (
                np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(self.official_key_embeddings, axis=1)
            )
            
            # Get top_k indices for each query
            top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
            
            # Convert to list of results for each query
            official_results = [
                [(self.official_keys[idx], float(similarities[query_idx][idx]))
                for idx in query_top_indices]
                for query_idx, query_top_indices in enumerate(top_indices)
            ]
            
            results = official_results
        
        if search_type in ['narration', 'both']:
            similarities = np.dot(query_embeddings, self.gt_narration_embeddings.T) / (
                np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(self.gt_narration_embeddings, axis=1)
            )
            
            top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
            
            narration_results = [
                [(self.gt_narrations[idx], float(similarities[query_idx][idx]))
                for idx in query_top_indices]
                for query_idx, query_top_indices in enumerate(top_indices)
            ]
            
            if search_type == 'both':
                # Merge and sort results for each query
                results = [
                    sorted(off + narr, key=lambda x: x[1], reverse=True)[:top_k]
                    for off, narr in zip(official_results, narration_results)
                ]
            else:
                results = narration_results
        
        return results
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the current state to HDF5 format for efficient storage of large arrays.
        
        Args:
            filepath (str): Path to save the state
        """
        with h5py.File(filepath + '.h5', 'w') as f:
            # Save embeddings as compressed datasets
            f.create_dataset('official_key_embeddings', 
                           data=self.official_key_embeddings,
                           compression='gzip', 
                           compression_opts=9)
            f.create_dataset('gt_narration_embeddings', 
                           data=self.gt_narration_embeddings,
                           compression='gzip', 
                           compression_opts=9)
            
            # Save text data separately in JSON
            text_data = {
                'official_keys': self.official_keys,
                'gt_narrations': self.gt_narrations
            }
            
            with open(filepath + '.json', 'w') as tf:
                json.dump(text_data, tf)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load state from HDF5 and JSON files.
        
        Args:
            filepath (str): Path to load the state from
        """
        with h5py.File(filepath + '.h5', 'r') as f:
            self.official_key_embeddings = f['official_key_embeddings'][:]
            self.gt_narration_embeddings = f['gt_narration_embeddings'][:]
            
        text_filepath = filepath + '.json'
        with open(text_filepath, 'r') as f:
            text_data = json.load(f)
            self.official_keys = text_data['official_keys']
            self.gt_narrations = text_data['gt_narrations']

def get_narrations_and_keys(anno_root):
    noun_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_noun_classes_v2.csv'))
    verb_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_verb_classes.csv'))    
    verb_maps = {} 
    noun_maps = {}
    for _, row in verb_classes_pd.iterrows():
        verb_maps[str(row['id'])] = row['key']
    for _, row in noun_classes_pd.iterrows():
        elements = row['key'].split(':')
        noun_maps[str(row['id'])] = ' '.join(elements[1:] + [elements[0]]) if len(elements) > 1 else row['key']
           
    narrations = []
    official_key_list = []
    for f in [
        os.path.join(anno_root, 'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:    
        csv_reader = csv.reader(open(f))
        next(csv_reader)
        for row in csv_reader:
            verb_id = str(row[10])
            noun_id = str(row[12])
            verb = verb_maps[verb_id]
            noun = noun_maps[noun_id]
            official_key = verb + ' ' + noun
            if official_key not in official_key_list:
                official_key_list.append(official_key)
                
            narrations.append(row[8])
    narrations = list(set(narrations))
    return list(set(narrations)), official_key_list
    
if __name__ == '__main__':
    # Load state
    retriever = SemanticRetriever()
    anno_root = '/data/shaokai/epic-kitchens-100-annotations/'
    gt_narrations,official_keys = get_narrations_and_keys(anno_root)
    #retriever.get_embeds(gt_narrations, official_keys)
    #retriever.save_to_file('embeddings')
    retriever.load_from_file('embeddings')
    
    print (retriever.find_similar('take spoon', search_type = 'official'))