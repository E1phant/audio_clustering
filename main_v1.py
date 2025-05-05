import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.ClassicalClustering import SongClusterer
from src.utils import write_json, print_playlist_dict


@hydra.main(version_base="1.2", config_path="configs", config_name="classical_config")
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

    hydra_cfg  = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    out_file = os.path.join(output_dir, "playlists_v1.json")
    write_json(out_file, playlists)
    print(f"Wrote {out_file!r}")

if __name__ == "__main__":
    main()
