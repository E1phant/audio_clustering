import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import librosa

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.utils import (
    get_audio_metadata,
    load_audio_file,
    pad_audio,
    majority_vote
)

class SongClusterer:
    def __init__(
            self, 
            data_path: Path|str,
            chunk_duration_s: float|int=5.0, 
            n_playlists: int=3,
            pad_last_chunk: bool=True,
            random_state: int=42,
            padding_threshold: float|int=0.05,
            n_components: int=20
    ):
        
        if not isinstance(data_path, (str, Path)):
            raise TypeError(f"'data_path' must be 'str' or 'Path', got {type(data_path).__name__!r}")
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path!r}")
        
        if not isinstance(chunk_duration_s, (int, float)):
            raise TypeError(f"'chunk_duration_s' must be 'int' or 'float', got {type(chunk_duration_s).__name__!r}")
        if chunk_duration_s <= 0:
            raise ValueError(f"'chunk_duration_s' must be > 0, got '{chunk_duration_s}'")
        
        if not isinstance(n_playlists, int):
            raise TypeError(f"'n_playlists' must be 'int', got {type(n_playlists).__name__!r}")
        if n_playlists < 1:
            raise ValueError(f"'n_playlists' must be ≥ 1, got '{n_playlists}'")
        
        if not isinstance(pad_last_chunk, bool):
            raise TypeError(f"'pad_last_chunk' must be 'bool', got {type(pad_last_chunk).__name__!r}")
        
        if not isinstance(random_state, int):
            raise TypeError(f"'random_state' must be 'int', got {type(random_state).__name__!r}")
        
        if not isinstance(padding_threshold, (int, float)):
            raise TypeError(f"'padding_threshold' must be 'int' or 'float', got {type(padding_threshold).__name__!r}")
        if not (0 <= padding_threshold <= 1):
            raise ValueError(f"'padding_threshold' must be between 0 and 1, got '{padding_threshold}'")
        
        if not isinstance(n_components, int):
            raise TypeError(f"'n_components' must be 'int', got {type(n_components).__name__!r}")
        if n_components < 1:
            raise ValueError(f"'n_components' must be ≥ 1, got '{n_components}'")

        self.chunk_duration_s = chunk_duration_s
        self.n_playlists = n_playlists
        self.data_path = data_path

        self.pad_last_chunk = pad_last_chunk
        self.padding_threshold = padding_threshold

        self.n_components = n_components

        self.random_state = random_state
        self.supported_formats = ('.mp3', '.wav', '.ogg')


    def _extract_features_chunk(self, chunk, sr):
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_var = np.var(mfccs, axis=1)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr))

        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr))

        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr))

        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(chunk))

        tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr)

        energy = np.mean(np.square(chunk))

        chroma = librosa.feature.chroma_stft(y=chunk, sr=sr)

        chroma_mean = np.mean(chroma, axis=1)

        features = np.hstack([
            mfcc_mean, 
            mfcc_var,
            spectral_centroid, spectral_bandwidth, spectral_rolloff,
            zero_crossing_rate,
            tempo,
            energy,
            chroma_mean
        ])
        return features

    def _extract_features_from_audio(self, audio_path, max_num_chunks, sr, pad_last_chunk):
        try:
            y = load_audio_file(audio_path, target_sr=sr)
            chunk_size = int(self.chunk_duration_s * sr)
            features_chunks = []

            for start in range(0, len(y) - chunk_size + 1, chunk_size):

                if max_num_chunks is not None and len(features_chunks) >= max_num_chunks:
                    break
                chunk = y[start:start+chunk_size]
                feat = self._extract_features_chunk(chunk, sr)
                features_chunks.append(feat)
            
            if pad_last_chunk:
                if max_num_chunks is not None and len(features_chunks) < max_num_chunks:
                    padded_last_chunk = pad_audio(y[chunk_size*len(features_chunks): len(y)], chunk_size)

                    pad_len = chunk_size - (len(y) - chunk_size*len(features_chunks))

                    if pad_len/chunk_size < self.padding_threshold:
                        feat = self._extract_features_chunk(padded_last_chunk, sr)
                        features_chunks.append(feat)

            return features_chunks
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []

    def cluster_songs(self):
        files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.lower().endswith(self.supported_formats)]
        print(f"Found {len(files)} audio files.")

        tqdm.pandas()
        try:
            audio_metadata = pd.DataFrame(pd.Series(files).progress_apply(get_audio_metadata).to_list())

            audio_length_q = audio_metadata['duration'].quantile(0.25)
            max_num_chunks = round(audio_length_q / self.chunk_duration_s)
            
            sr_mean = int(audio_metadata['sample_rate'].mean())
        except Exception as e:
            print(f"Error processing audio files {files}: {e}")
            return {}

        all_chunk_features = []
        chunk_song_map = []
        pad_last_chunk = self.pad_last_chunk

        for file in tqdm(files, desc="Extracting features"):
            features_chunks = self._extract_features_from_audio(file, max_num_chunks, sr_mean, pad_last_chunk)
            if features_chunks:
                for feat in features_chunks:
                    all_chunk_features.append(feat)
                    chunk_song_map.append(file)
                    

        if not all_chunk_features:
            print("No features extracted from any files.")
            return {}
        
        all_chunk_features = np.array(all_chunk_features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_chunk_features)

        pca = PCA(n_components=self.n_components)
        reduced_features = pca.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters=self.n_playlists, random_state=self.random_state)
        chunk_labels = kmeans.fit_predict(reduced_features)

        song_labels = {}
        for song, label in zip(chunk_song_map, chunk_labels):
            song_labels.setdefault(song, []).append(label)

        playlists = {i: [] for i in range(self.n_playlists)}

        for song, labels_list in song_labels.items():
            majority_label = majority_vote(labels_list)
            playlists[majority_label].append(song)
        return playlists