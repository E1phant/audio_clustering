import os
import hydra
from omegaconf import DictConfig

from src.ClassicalClustering import SongClusterer
from src.utils import write_json, print_playlist_dict


@hydra.main(version_base=None, config_path="configs", config_name="classical_config")
def main(cfg: DictConfig):
    clusterer = SongClusterer(
        data_path=cfg.path,
        chunk_duration_s=cfg.chunk_duration_s,
        n_playlists=cfg.n,
        pad_last_chunk=cfg.pad_last_chunk,
        random_state=cfg.random_state,
        padding_threshold=cfg.padding_threshold,
        n_components=cfg.n_components,
    )
    
    playlists = clusterer.cluster_songs()
    print_playlist_dict(playlists)

    out_file = os.path.join(cfg.path, "playlists_v1.json")
    write_json(out_file, playlists)
    print(f"Wrote {out_file!r}")

if __name__ == "__main__":
    main()
