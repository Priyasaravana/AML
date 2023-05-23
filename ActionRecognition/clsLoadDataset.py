import os
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import gc

def getDataLoader():
    if os.path.exists("HMDBDataset.pickle"):
        with open("HMDBDataset.pickle", "rb") as pkl:
            return pickle.load(pkl)
        
    ds = HMDBDs(
        "data\HMDB_simp")
    with open("HMDBDataset.pickle", "wb") as pkl:
        pickle.dump(ds, pkl)
    return ds

class HMDBDs(Dataset):
    imgTransform = T.Compose(
        [T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])

    def __init__(self, dataset_dir):
        self.labels = []
        self.instances = []
        self.readVideos(dataset_dir)
        self.instances = torch.from_numpy(np.array(self.instances))
        self.labels = torch.from_numpy(np.array(self.labels))

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]

    def readVideos(self, dataset_dir):
        self.labels_ref = list(os.listdir(dataset_dir))
        for action in tqdm(self.labels_ref):
            # print(f"Loading Action: {action}")
            action_path = os.path.join(dataset_dir, action)
            for video_folder in os.listdir(action_path):
                gc.collect()
                video_folder_path = os.path.join(action_path, video_folder)
                images_paths = os.listdir(video_folder_path)                
                frame_indices = np.linspace(0, len(images_paths)-1, num=8).astype(np.int64)
                self.instances.append(np.array([np.array(self.imgTransform(np.asarray(Image.open(os.path.join(video_folder_path, images_paths[img_idx] ))))) for img_idx in frame_indices]))
                self.labels.append(self.labels_ref.index(action))

    def parseVideo(self, frames):
        resized = np.array(
            [np.array(self.imgTransform(frame)) for frame in frames])
        result = self.sample_frame_indices(8, 3, len(resized))
        return np.array([resized[idx] for idx in result])

    # def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
    #     converted_len = int(clip_len * frame_sample_rate)
    #     # print(f"Converted Len: {converted_len}, Seg Len: {seg_len}")
    #     end_idx = np.random.randint(converted_len, seg_len)
    #     start_idx = end_idx - converted_len
    #     indices = np.linspace(start_idx, end_idx, num=clip_len)
    #     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    #     return indices


