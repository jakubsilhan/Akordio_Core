import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import pyrubberband as pyrb
import os, librosa, shutil, io, torch
from sklearn.model_selection import KFold, train_test_split
from ..Classes.NetConfig import Config, load_config

class Preprocessor():
    def __init__(self, config: Config):
        self.config = config

    def process_all_data(self) -> None:
        '''
        Goes through all specified data and processes them into a dataset according to configuration
        '''
        if os.path.exists(self.config.data.preprocessed_dir):
            shutil.rmtree(self.config.data.preprocessed_dir)
        os.makedirs(self.config.data.preprocessed_dir)
        for dataset in tqdm(self.config.data.datasets, desc="Processing datasets"):
            self.process_dataset(dataset)
        shutil.copy2("config.yaml", self.config.data.preprocessed_dir)

    def process_dataset(self, dataset: str) -> None:
        '''
        Goes through all songs in a dataset a processes them
        '''
        dataset_path = os.path.join(self.config.data.dataset_dir, dataset)
        all_files = []
        for path, _, files in os.walk(os.path.join(dataset_path, "Audio")):
            for filename in files:
                all_files.append((path, filename))

        all_files.sort()

        # Train/Test split
        train_files, test_files = train_test_split(
            all_files, 
            test_size=self.config.data.preprocess.test_split, 
            random_state=self.config.base.random_seed,
            shuffle=True
        )

        # Test set
        test_path = os.path.join(self.config.data.preprocessed_dir, "test", "0")
        os.makedirs(test_path, exist_ok=True)
        for path, filename in tqdm(test_files, desc="Processing test set"):
            self.process_song(path, filename, test_path)

        # Training set with K-Fold
        kf = KFold(self.config.data.preprocess.num_splits, shuffle=True, random_state=self.config.base.random_seed)
        train_path = os.path.join(self.config.data.preprocessed_dir, "train")
        os.makedirs(train_path, exist_ok=True)
        fold_pbar = tqdm(total=self.config.data.preprocess.num_splits, desc="Generating folds")
        for fold, (_, fold_indices) in enumerate(kf.split(train_files)):
            fold_path = os.path.join(train_path, str(fold))
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            fold_files = [train_files[idx] for idx in fold_indices]
            for path, filename in tqdm(fold_files, desc="Processing fold songs"):
                self.process_song(path, filename, fold_path)
            fold_pbar.update(1)
        fold_pbar.close()

    def process_song(self, path: str, filename: str, out_path: str) -> None:
        '''
        Processes a song and applies pitch shifting aswell if needed
        '''
        audio_path = os.path.join(path, filename)
        y, sr = librosa.load(audio_path, sr=self.config.data.preprocess.sampling_rate)
        intervals = self.load_annotation(path, filename)
        shifts = range(self.config.data.preprocess.pitch_shift_start, self.config.data.preprocess.pitch_shift_end+1)
        for shift_factor in shifts:
            if shift_factor != 0:
                y_shifted = pyrb.pitch_shift(y, sr=sr, n_steps=shift_factor)
            else:
                y_shifted = y
            features, times = self.process_features(y_shifted)
            intervals_shifted = self.shift_annotation(intervals, shift_factor)
            labels = self.assign_labels_to_times(times, intervals_shifted)

            song_df = pd.concat([
              pd.DataFrame({"timestamp": times}),
              pd.DataFrame(features),
              pd.DataFrame({"chord": labels})
            ], axis=1)

            # Saving
            save_base = filename.replace(".mp3", "").split("_-_")[-1]
            self.save_fragments(song_df, save_base, out_path, shift_factor)

    def process_audio(self, y: np.ndarray) -> list[torch.Tensor]:
        """
        Processes audio into features according to the config
        """
        
        # Extract features
        features,_ = self.process_features(y)

        # Split into fragments
        fragment_size = self.config.data.preprocess.fragment_size
        fragments = []

        # Return with no fragmenting
        if fragment_size == 0:
            fragments.append(torch.tensor(features, dtype=torch.float32))
            return fragments
        
        # Fragment
        for start in range(0, len(features), fragment_size):
            fragment = features[start:start+fragment_size]
            fragments.append(torch.tensor(fragment, dtype=torch.float32))
        
        return fragments

    def process_features(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Processes audio into a log CQT or chromagram
        '''
        if self.config.data.preprocess.pcp.enabled:
            features = librosa.feature.chroma_cqt(y=y, sr=self.config.data.preprocess.sampling_rate, bins_per_octave=self.config.data.preprocess.bins_per_octave, hop_length=self.config.data.preprocess.hop_length, n_chroma=self.config.data.preprocess.pcp.bins, n_octaves=self.config.data.preprocess.pcp.octaves)
        else:
            cqt = np.abs(librosa.cqt(y, sr=self.config.data.preprocess.sampling_rate, bins_per_octave=self.config.data.preprocess.bins_per_octave,n_bins=self.config.data.preprocess.cqt_bins, hop_length=self.config.data.preprocess.hop_length))
            features = cqt
    
        features = features.T
        times = librosa.frames_to_time(np.arange(features.shape[0]), sr=self.config.data.preprocess.sampling_rate, hop_length=self.config.data.preprocess.hop_length)

        return features, times

    def load_annotation(self, path: str, filename: str) -> list[tuple[float, float, str]]:
        '''
        Loads annotation into a list of tuples
        '''
        path = path.replace("Audio", "Chords")
        filename = filename.replace("mp3", "lab")
        filepath = os.path.join(path, filename)
        intervals = []
        with open(filepath) as f:
            for line in f:
                start, end, chord = line.strip().split()
                intervals.append((float(start), float(end), chord))

        return intervals
    
    def shift_annotation(self, intervals: list[tuple[float, float, str]], shift_factor) -> list:
        '''
        Shift labels in the annotations according to the shift factor
        '''
        shifted = []
        for start, end, label in intervals:
            new_label = self.shift_root(label, shift_factor)
            shifted.append((start, end, new_label))
        return shifted

    def shift_root(self, chord: str, semitone_shift: int) -> str:
        note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        if 'N' in chord:
            return 'N'
        
        extensions = ""
        bass_interval = ""

        # Bass
        if '/' in chord:
            chord, bass_interval = chord.split("/", 1)
            bass_interval = "/" + bass_interval
        
        # Extensions
        if '(' in chord:
            extensions = chord[chord.find('('):]
            chord = chord[:chord.find('(')]

        # Parse root and quality
        if ':' in chord:
            root, quality = chord.split(':', 1)
        elif len(chord) >= 2 and chord[1] in ['b', '#']:
            root = chord[:2]
            quality = chord[2:] if len(chord) > 2 else 'maj'
        else:
            root = chord[:1]
            quality = chord[1:] if len(chord) > 1 else 'maj'
        
        # Shift only the root
        root_note = self.normalize_note(root)
        idx = note_list.index(root_note)
        new_idx = (idx + semitone_shift) % 12
        shifted_root = note_list[new_idx]
        
        # Reconstruct chord
        result = f"{shifted_root}:{quality}{extensions}{bass_interval}"
        
        return result

    # Utils
    def save_fragments(self, song_df: pd.DataFrame, base_name: str, out_path: str, shift_factor: int) -> None:
        '''
        Splits the song dataframe into fixed-size fragments (in frames) and saves them as individual npz files
        '''
        if self.config.data.preprocess.pcp.enabled:
            input_dim = self.config.data.preprocess.pcp.bins
        else:
            input_dim = self.config.data.preprocess.cqt_bins

        # Full song mode
        if self.config.data.preprocess.fragment_size <= 0:
            # Extract into numpy arrays
            timestamps = song_df.iloc[:, 0].values.astype(np.float32)
            X = song_df.iloc[:, 1:1 + input_dim].values.astype(np.float32) # skip timestamp
            y = song_df["chord"].values.astype(str)

            # Prepare pathing
            song_filename = f"{base_name}_shift{shift_factor:02d}.npz"
            song_path = os.path.join(out_path, song_filename)
            os.makedirs(os.path.dirname(song_path), exist_ok=True)

            # Save into npz
            np.savez_compressed(song_path, timestamps=timestamps, X=X, y=y) # type: ignore
            return

        # Fragmenting mode
        num_rows = len(song_df)
        hop_size = int(self.config.data.preprocess.fragment_size * self.config.data.preprocess.fragment_hop)
        for start in range(0, num_rows, hop_size):
            fragment = song_df.iloc[start:start + self.config.data.preprocess.fragment_size]

            if len(fragment) < self.config.data.preprocess.fragment_size:
                continue

            X = fragment.iloc[:, 1:1 + input_dim].values.astype(np.float32)
            y = fragment["chord"].values.astype(str)
            timestamps = fragment.iloc[:, 0].values.astype(np.float32)

            # Prepare path
            frag_filename = f"{base_name}_shift{shift_factor:02d}_frag{start//hop_size:04d}.npz"
            frag_path = os.path.join(out_path, frag_filename)
            os.makedirs(os.path.dirname(frag_path), exist_ok=True)

            # Save fragment
            np.savez_compressed(frag_path, timestamps=timestamps, X=X, y=y) # type: ignore



    def assign_labels_to_times(self, times, intervals) -> np.ndarray:
        '''
        Creates an array of chord aligned to specific timings
        '''
        if len(intervals) == 0:
            return np.full(len(times), "N", dtype=object)
        
        # Vectorization
        starts = np.array([i[0] for i in intervals])
        ends = np.array([i[1] for i in intervals])
        chords = np.array([i[2] for i in intervals])
        
        # Find fitting timestamps
        indices = np.searchsorted(starts, times, side='right') - 1
        
        # Check if times fall within <start, end) of their assigned interval
        valid = (indices >= 0) & (times < ends[np.clip(indices, 0, len(ends)-1)])
        
        # Assign labels
        labels = np.where(valid, chords[indices], "N")
        
        return labels


    def normalize_note(self, note: str) -> str:
        '''
        Normalizes flats
        '''
        flat_to_sharp = {
            'Cb': 'B',
            'Db': 'C#',
            'Eb': 'D#',
            'Fb': 'E',
            'Gb': 'F#',
            'Ab': 'G#',
            'Bb': 'A#',
            'E#': 'F',
            'B#': 'C'
        }
        if note in flat_to_sharp:
            return flat_to_sharp[note]
        return note
    

# if __name__ == "__main__":
#     config = load_config("config.yaml")
#     preprocessing = Preprocess(config)
#     preprocessing.process_all_data()
