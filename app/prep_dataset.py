import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

with open('WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

video_data = []

for entry in wlasl_data:
    gloss = entry['gloss']
    instances = entry['instances']
    
    for instance in instances:
        video_id = instance['video_id']
        signer_id = instance.get('signer_id', None)
        split = instance.get('split', None)
        
        video_path = os.path.join('videos', f'{video_id}.mp4')
        
        if os.path.exists(video_path):
            video_data.append({
                'video_id': video_id,
                'gloss': gloss,
                'signer_id': signer_id,
                'split': split,
                'video_path': video_path
            })

df = pd.DataFrame(video_data)

print(f"Total videos found: {len(df)}")
print(f"Total unique signs: {df['gloss'].nunique()}")
print(f"Missing videos: {len(wlasl_data) - len(df)}")

df.to_csv('wlasl_metadata.csv', index=False)

if df['split'].isna().all():
    from sklearn.model_selection import train_test_split
    
    train_val_data = []
    test_data = []
    
    for gloss, group in df.groupby('gloss'):
        # 80% train+val, 20% test
        group_train_val, group_test = train_test_split(
            group, test_size=0.2, random_state=42)
        
        train_val_data.append(group_train_val)
        test_data.append(group_test)
    
    train_val_df = pd.concat(train_val_data)
    test_df = pd.concat(test_data)
    
    train_data = []
    val_data = []
    
    for gloss, group in train_val_df.groupby('gloss'):
        # 75% train, 25% validation (overall: 60% train, 20% val, 20% test)
        group_train, group_val = train_test_split(
            group, test_size=0.25, random_state=42)
        
        train_data.append(group_train)
        val_data.append(group_val)
    
    train_df = pd.concat(train_data)
    val_df = pd.concat(val_data)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv('wlasl_with_splits.csv', index=False)
    
    print(f"Train set: {len(train_df)} videos")
    print(f"Validation set: {len(val_df)} videos")
    print(f"Test set: {len(test_df)} videos")