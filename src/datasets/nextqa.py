import pandas as pd
import os

class NExTQA:
    def __init__(self, split='val', root_dir='data/nextqa'):
        """
        NExT-QA Dataset Loader.
        Args:
            split (str): 'train', 'val', or 'test'.
            root_dir (str): Path to data directory containing csv files and 'videos' subdir.
        """
        self.split = split
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, 'videos')
        self.csv_path = os.path.join(root_dir, f'{split}.csv')
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Annotation file not found: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        video_id = str(row['video'])
        # video_path = os.path.join(self.video_dir, f"{video_id}.mp4") 
        # Note: NExT-QA videos usually have arbitrary filenames, but user will need to organize them
        # or we assume they are named by ID. For now assuming ID.mp4.
        
        # NExT-QA provides multiple choice options a0-a4
        candidates = [str(row[f'a{i}']) for i in range(5)]
        
        sample = {
            'index': idx,
            'video_id': video_id,
            'video_path': os.path.join(self.video_dir, f"{video_id}.mp4"),
            'question': str(row['question']),
            'candidates': candidates,
            'answer_idx': int(row['answer']),
            'type': str(row['type']),
            'qid': str(row['qid']),
            'width': int(row['width']),
            'height': int(row['height'])
        }
        
        return sample

    def get_video_path(self, video_id):
        return os.path.join(self.video_dir, f"{video_id}.mp4")
