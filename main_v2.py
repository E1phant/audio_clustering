import os
import hydra
from omegaconf import DictConfig

from transformers import ClapProcessor, ClapModel
from src.DeepClustering import DeepSongClusterer
from src.utils import write_json, print_playlist_dict

@hydra.main(version_base=None, config_path="configs", config_name="deep_config")
def main(cfg: DictConfig):
    print("Loading processor & model…")
    processor = ClapProcessor.from_pretrained(cfg.processor_ckpt)
    model = ClapModel.from_pretrained(cfg.model_ckpt)

    print("Initializing clusterer…")
    clusterer = DeepSongClusterer(
        data_path=cfg.path,
        n_playlists=cfg.n,
        strategy=cfg.strategy,
        chunk_duration=cfg.chunk_duration,
        sr=cfg.sr,
        processor=processor,
        model=model,
        device=cfg.device,
        random_state=cfg.random_state,
        use_pca=cfg.use_pca,
        pca_n_components=cfg.pca_n_components,
        use_mid_layers=cfg.use_mid_layers,
        start_mid_layer_idx=cfg.start_mid_layer_idx,
        end_mid_layer_idx=cfg.end_mid_layer_idx
    )

    print("Clustering…")
    playlists = clusterer.cluster_songs()
    print_playlist_dict(playlists)

    out = os.path.join(cfg.path, "playlists_v2.json")
    write_json(out, playlists)
    print(f"Wrote {out!r}")

if __name__ == "__main__":
    main()
