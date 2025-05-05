import os
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
from transformers import AutoProcessor, AutoModelForPreTraining
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from sklearn.decomposition import PCA

from src.utils import (
    load_audio_file,
    fixed_chunk_indices,
    pad_audio,
    majority_vote
)


class DeepSongClusterer:
    def __init__(
        self,
        data_path: str,
        n_playlists: int,
        strategy: str="chunk_average",
        chunk_duration: float=5.0,
        sr: int=48000,

        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused"),
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused"),

        device: str=None,
        random_state: int=42,
        use_pca: bool=False,
        pca_n_components: int=20,
        use_mid_layers: bool=False,
        start_mid_layer_idx: int=6,
        end_mid_layer_idx: int=9
    ):
        strategies = ["chunk_average", "chunk_level"]

        if not isinstance(data_path, (str, Path)):
            raise TypeError(f"'data_path' should be 'str' or 'Path', but got {type(data_path).__name__!r}")
        data_path = Path(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: '{data_path!r}")
        
        if not isinstance(n_playlists, int):
            raise TypeError(f"'n_playlists' should be 'int', but got {type(n_playlists).__name__!r}")
        if n_playlists < 0:
            raise ValueError(f"'n_playlists' must be non-negative, got '{n_playlists}'")
        
        if not isinstance(strategy, str):
            raise TypeError(f"'strategy' should be 'str', but got {type(strategy).__name__!r}")
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy!r}. Expected one of {strategies}")
        
        if not isinstance(chunk_duration, (float, int)):
            raise TypeError(f"'chunk_duration' should be 'float' or 'int', but got {type(chunk_duration).__name__!r}")
        if chunk_duration < 0:
            raise ValueError(f"'chunk_duration' must be non-negative, got '{chunk_duration}'")
        
        if not isinstance(sr, int):
            raise TypeError(f"'sr' should be 'int', but got {type(n_playlists).__name__!r}")
        if sr < 0:
            raise ValueError(f"'sr' must be non-negative, got {sr}")

        if not isinstance(processor, (ClapProcessor, AutoProcessor)):
            raise TypeError(f"'processor' must be a ClapProcessor or AutoProcessor, but got {type(processor).__name__!r}")
        
        if not isinstance(model, (ClapModel, AutoModelForPreTraining)):
            raise TypeError(f"'model' must be a ClapProcessor or AutoProcessor, but got {type(model).__name__!r}")
        
        model_type = model.config.model_type

        if model_type not in ["clap", "wav2vec2"]:
            raise ValueError( f"Unsupported model_type: {model_type!r}. Expected one of ['clap', 'wav2vec2']")
        if use_mid_layers and model_type != "wav2vec2":
            raise ValueError(f"use_mid_layers=True only supported for 'wav2vec2' models, but model_type is {model_type!r}")
        
        if not isinstance(use_pca, bool):
            raise TypeError(f"'use_pca' must be bool, got {type(use_pca).__name__!r}")
        if not isinstance(pca_n_components, int):
            raise TypeError(f"'pca_n_components' must be int, got {type(pca_n_components).__name__!r}")
        if pca_n_components <= 0:
            raise ValueError(f"'pca_n_components' must be > 0, got {pca_n_components}")
        if not isinstance(random_state, int):
            raise TypeError(f"'random_state' must be int, got {type(random_state).__name__!r}")
        

        

        self.data_path = data_path
        self.n_playlists = n_playlists
        self.strategy = strategy

        self.chunk_duration = chunk_duration
        self.target_sr = sr

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor
        self.model = model.to(self.device).eval()
        self.model_type = model_type

        self.use_pca = use_pca
        self.use_mid_layers = use_mid_layers
        
        self.pca_n_components = pca_n_components
        self.random_state = random_state

        self.start_mid_layer_idx = 6 if start_mid_layer_idx is None else start_mid_layer_idx
        self.end_mid_layer_idx = 9 if end_mid_layer_idx is None else end_mid_layer_idx

        

    def extract_features_from_array(self, y):
        inputs = self.processor(audios=np.asarray(y, dtype=np.float64), sampling_rate=self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)
        
        if self.model_type == "clap":
            embedding = outputs
        
        elif self.model_type == "wav2vec2":
            if self.use_mid_layers:
                mid_hidden_states = outputs.hidden_states[self.start_mid_layer_idx:self.end_mid_layer_idx+1]
                avg_mid = torch.stack(mid_hidden_states, dim=0).mean(dim=0)
                embedding = avg_mid.mean(dim=1)
            
            else:
                embedding = outputs

        

        return embedding.squeeze().cpu().numpy()



    def _extract_song_embeddings(self):
        supported_formats = (".mp3", ".wav", ".ogg")
        files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.lower().endswith(supported_formats)]

        song_embeddings = {}
        chunk_features = []
        chunk_song_map = []

        for file in tqdm(files, desc="Processing songs"):
            try:
                y = load_audio_file(file, self.target_sr)
                indices = fixed_chunk_indices(len(y), self.chunk_duration, self.target_sr)

                if self.strategy == "chunk_average":
                    features = []
                    for start, end in indices:
                        chunk = pad_audio(y[start:end], int(self.chunk_duration * self.target_sr))
                        features.append(self.extract_features_from_array(chunk))
                    if features:
                        song_embeddings[file] = np.mean(features, axis=0)

                elif self.strategy == "chunk_level":
                    for start, end in indices:
                        chunk = pad_audio(y[start:end], int(self.chunk_duration * self.target_sr))
                        chunk_features.append(self.extract_features_from_array(chunk))
                        chunk_song_map.append(file)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        if self.strategy == "chunk_average":
            files_embedded = [f for f, emb in song_embeddings.items() if emb is not None]
            if not files_embedded:
                return None, []
            features_matrix = np.array([song_embeddings[f] for f in files_embedded])

        elif self.strategy == "chunk_level":
            if not chunk_features:
                return None, []
            files_embedded = chunk_song_map
            features_matrix = np.array(chunk_features)

        return features_matrix, files_embedded
    


    def _cluster_embeddings(self, features_matrix, files_embedded):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)

        if self.use_pca:
            pca_model = PCA(n_components=self.pca_n_components)
            features_reduced = pca_model.fit_transform(features_scaled)
        else:
            features_reduced = features_scaled

        kmeans = KMeans(n_clusters=self.n_playlists, random_state=self.random_state)
        labels = kmeans.fit_predict(features_reduced)

        playlists = defaultdict(list)

        if self.strategy == "chunk_average":
            for file, label in zip(files_embedded, labels):
                playlists[label].append(file)

        elif self.strategy == "chunk_level":
            song_label_mapping = defaultdict(list)
            for file, label in zip(files_embedded, labels):
                song_label_mapping[file].append(label)
            for file, label_list in song_label_mapping.items():
                playlists[int(majority_vote(label_list))].append(file)

        return dict(playlists)



    def cluster_songs(self):
        features_matrix, files_embedded = self._extract_song_embeddings()

        if features_matrix is None or not files_embedded:
            return {}

        return self._cluster_embeddings(features_matrix, files_embedded)