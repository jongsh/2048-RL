import yaml

register_path = "configs/register.yaml"
with open(register_path, "r") as file:
    register = yaml.safe_load(file)

config_path = "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


def load_config():
    """Load full configuration from the config file"""
    ret_config = {}
    for key, value in config["public"].items():
        if key not in ret_config:
            ret_config[key] = value

    ret_config["env"] = config["env"][config["public"]["env"]]
    ret_config["agent"] = config["agent"][config["public"]["agent"]]
    ret_config["model"] = config["model"][config["public"]["model"]]
    # TODO

    return ret_config


def load_single_config(type, config_name):
    """Load a single specific configuration based on type and name"""
    if type not in register:
        raise ValueError(f"Type '{type}' not found in register.yaml")
    if config_name not in register[type]:
        raise ValueError(f"Config name '{config_name}' not found under type '{type}' in register.yaml")

    ret_config = config[type][config_name]
    for key, value in config["public"].items():
        if key not in ret_config:
            ret_config[key] = value

    return ret_config
