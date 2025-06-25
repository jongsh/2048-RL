import yaml

config_name_dict = {"game2048": "config/game2048.yaml"}


def load_config(config_name):
    file_path = config_name_dict.get(config_name)
    if not file_path:
        raise ValueError(f"Unknown config name: {config_name}")

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config
