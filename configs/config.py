import threading
from omegaconf import OmegaConf


class Configuration:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path="configs/config.yaml", register_path="configs/register.yaml", cli_args=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Configuration, cls).__new__(cls)
                    cls._instance._init(config_path, register_path, cli_args)
        return cls._instance

    def _init(self, config_path, register_path, cli_args):
        self.config = OmegaConf.load(config_path)
        self.register = OmegaConf.load(register_path)
        # Override config with command-line arguments
        if cli_args:
            dotlist = [arg[2:] for arg in cli_args if arg.startswith("--")]
            cli_conf = OmegaConf.from_dotlist(dotlist)
            self.config = OmegaConf.merge(self.config, cli_conf)

        if self._validate():
            print("Configuration is valid.")

        self.ret_config = {}
        self.ret_config["public"] = self.config["public"]
        self.ret_config["env"] = self.config["env"][self.config["public"]["env"]]
        self.ret_config["agent"] = self.config["agent"][self.config["public"]["agent"]]
        self.ret_config["model"] = self.config["model"][self.config["public"]["model"]]
        self.ret_config["trainer"] = self.config["trainer"][self.config["public"]["trainer"]]

    def get_config(self, key):
        """Return the validated configuration"""
        return self.ret_config[key] if key in self.ret_config else None

    def save_config(self, dir_path):
        """Save the configuration to the specified directory"""
        save_path = f"{dir_path}/config.yaml"
        OmegaConf.save(self.config, save_path)

    def to_string(self):
        """Return the configuration as a formatted string"""

        def format_value(v, indent=2):
            if isinstance(v, dict):
                lines = [""]
                for k2, v2 in v.items():
                    sub_str = format_value(v2, indent + 2)
                    lines.append(" " * indent + f"{k2}: {sub_str}")
                return "\n".join(lines)
            elif isinstance(v, list):
                return "[" + ", ".join(map(str, v)) + "]"
            else:
                return str(v)

        lines = ["=" * 10 + " Configuration Summary " + "=" * 10, ""]
        for section, content in self.ret_config.items():
            lines.append(f"[{section}]")
            lines.append(format_value(content, indent=2))
            lines.append("")
        lines.append("=" * 40)
        return "\n".join(lines)

    def _validate(self):
        public_config = self.config["public"]

        # Validate environment
        if public_config["env"] not in self.register["env"]:
            raise ValueError(f"Environment {public_config['env']} not registered.")

        # Validate agent
        if public_config["agent"] not in self.register["agent"]:
            raise ValueError(f"Agent {public_config['agent']} is not registered.")

        # Validate model
        if public_config["model"] not in self.register["model"]:
            raise ValueError(f"Model {public_config['model']} is not registered.")

        # Validate trainer
        if public_config["trainer"] not in self.register["trainer"]:
            raise ValueError(f"Trainer {public_config['trainer']} is not registered.")

        return True
