import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .NetConfig import Config
from Akordio_Core.Tools.Chords import Chords, Complexity
from typing import List

class SongDataset(Dataset):
    def __init__(self, paths: List[str], config: Config):
        """
        Initializes the dataset with song fragment file paths.
        Each song fragment is a made of tensors (X, y)
        X = sequence of feature vectors (chroma/cqt)
        y = sequence of labels (majmin/majmin7/complex)
        """
        self.config = config
        self.paths = paths
        self.chord_tool = Chords()
        self.complexity = self._get_complexity()

    def _get_complexity(self) -> Complexity:
        """
        Get chord complexity from config
        """
        match self.config.train.model_complexity:
            case "complex":
                return Complexity.COMPLEX
            case "majmin7":
                return Complexity.MAJMIN7
            case _:
                return Complexity.MAJMIN

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieves the requested song fragment, loads the file and converts it to tensors (X,y) of features and labels.
        """
        # Get indexed path
        path = self.paths[idx]

        # Load song fragment
        data = np.load(path)

        X = torch.from_numpy(data["X"]).float()
        
        # Encode according to prescribed complexity
        complexity_key = f"y_{self.config.train.model_complexity}".lower()
        y = torch.from_numpy(data[complexity_key]).long()

        # Optional CQT logarithmization
        if not self.config.data.preprocess.pcp.enabled:
            X = torch.log(X+1e-6)

        return X, y

# Collate function class
class PaddingCollate:
    """Collate function that pads sequences"""
    def __init__(self, padding_index):
        self.padding_index = padding_index
    
    def __call__(self, batch):
        # Sort the batch by the length of X (descending order)
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Extract X and y from batch
        X_batch, y_batch = zip(*batch)
        
        # Pad feature sequences
        X_batch_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)  # type: ignore
        
        # Pad labels
        y_batch_padded = pad_sequence(y_batch, batch_first=True, padding_value=self.padding_index) # type: ignore
        
        return X_batch_padded, y_batch_padded


# Factory function
def make_collate_fn(padding_index):
    return PaddingCollate(padding_index)