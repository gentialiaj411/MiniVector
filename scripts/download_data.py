"""
Download sample text data for vector database.
Uses a smaller, faster dataset perfect for testing.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_text_data(num_samples=100000, output_path="data/raw/texts.json"):
    """
    Download text dataset - using AG News (fast, reliable).
    """
    print(f"Downloading {num_samples} text documents...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading AG News dataset (fast download)...")
    dataset = load_dataset(
        "ag_news",
        split=f"train[:{num_samples}]"
    )
    
    print(f"Processing {len(dataset)} documents...")
    
    documents = []
    for i, item in enumerate(tqdm(dataset)):
        text = item['text'].strip()
        
        if len(text) < 50:
            continue
        text = text[:500]
        
        categories = ['World', 'Sports', 'Business', 'Technology']
        category = categories[item['label']]
        
        documents.append({
            'id': f'doc_{i}',
            'text': text,
            'title': f"{category} Article {i}",
            'category': category
        })

    print(f"\nSaving {len(documents)} documents to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully saved {len(documents)} documents")
    print(f"✓ File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    if documents:
        print("\nSample document:")
        print(f"ID: {documents[0]['id']}")
        print(f"Category: {documents[0]['category']}")
        print(f"Text: {documents[0]['text'][:150]}...")

if __name__ == "__main__":
    download_text_data(num_samples=100000)