import json
from typing import Any, Mapping


def load_json(path: str) -> Mapping[str, Any]:
    with open(path, "r") as read_file:
        loaded_dict = json.load(read_file)
    return loaded_dict


def write_json(path, data):
    data_cleaned = {int(k): v for k, v in data.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_cleaned, f, ensure_ascii=False, indent=4)


def print_playlist_dict(playlist_dict):
    print("Clustered Playlists:")
    for cluster_id, songs in sorted(playlist_dict.items()):
        print(f"\nPlaylist {cluster_id}:")
        for song_path in songs:
            print(f" - {song_path}")