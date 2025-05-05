import numpy as np

import librosa
import torchaudio

from collections import Counter

def load_audio_file(audio_path, target_sr=None):
    y, sr = librosa.load(audio_path, sr=None)
    if target_sr is None:
        return y
    elif sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def get_audio_metadata(file_path: str):
    metadata = torchaudio.info(file_path)

    sample_rate = metadata.sample_rate
    num_channels = metadata.num_channels
    num_frames = metadata.num_frames
    duration = num_frames / sample_rate if sample_rate else None
    
    bit_depth = getattr(metadata, "bits_per_sample", None)
    encoding = getattr(metadata, "encoding", None) 
    
    return {
        "file_path": file_path,
        "sample_rate": sample_rate,
        "duration": duration,
        "num_channels": num_channels,
        "bit_depth": bit_depth,
        "encoding": encoding,
    }



def fixed_chunk_indices(y_len, chunk_duration, sr):
    chunk_size = int(chunk_duration * sr)
    return [(start, start + chunk_size) for start in range(0, y_len, chunk_size)]



def pad_audio(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio



def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]
