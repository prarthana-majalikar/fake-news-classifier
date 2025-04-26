import pandas as pd

def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels (1 for real, 0 for fake)
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine datasets
    df = pd.concat([fake_df[['title', 'text', 'label']], true_df[['title', 'text', 'label']]])
    
    # Shuffle data for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

#testing and checking shape
if __name__ == "__main__":
    fake_path = 'data/Fake.csv'
    true_path = 'data/True.csv'
    data = load_data(fake_path, true_path)
    print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    print(data.head())
