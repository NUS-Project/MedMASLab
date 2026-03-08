from __future__ import annotations

from methods.maslab_runtime_config import build_general_config
from .dylan_main import DyLAN_Main
from pathlib import Path
import yaml
def load_config(file_path: str) -> dict:
    """
    Load YAML configuration from a file.

    Args:
    file_path (str): Path to the YAML configuration file.

    Returns:
    dict: Dictionary of configuration parameters.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def dylan_infer_medqa(question, root_path, model_info, img_paths=None,batch_manager=None):
    # config_path = str(Path(root_path) / 'methods' / 'dylan' / 'configs' / 'config_main.yaml')
    # general_config = build_general_config(root_path, model_info)
    mas = DyLAN_Main(model_name="dylan",batch_manager=batch_manager)

    result, current_config = mas.inference({"query": question, "img_paths": img_paths})

    response = (result or {}).get("response", "")
    print(f"dylan_response:{response}")
    token_stats = mas.get_token_stats()
    return str(response).strip(), token_stats, current_config
