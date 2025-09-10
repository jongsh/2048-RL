import yaml
import threading


class Configuration:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path="configs/config.yaml", register_path="configs/register.yaml"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Configuration, cls).__new__(cls)
                    cls._instance._init(config_path, register_path)
        return cls._instance

    def _init(self, config_path, register_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        with open(register_path, "r") as file:
            self.register = yaml.safe_load(file)

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
        with open(save_path, "w") as file:
            yaml.dump(self.config, file)
    
    def to_string(self):
        """Return the configuration as a formatted string"""
        return yaml.dump(
        self.config,
        sort_keys=False,
        default_flow_style=False,  # 紧凑的 {key: value} 样式
        indent=2                  # 缩进保持好看
    )

    def _validate(self):
        public_config = self.config["public"]

        # Validate environment
        if public_config["env"] not in self.register["env"]:
            raise ValueError(f"Environment {public_config['env']} not registered.")

        # Validate agent
        if public_config["agent"] not in self.register["agent"]:
            raise ValueError(f"Agent {public_config['agent']} not registered.")

        # Validate model
        if public_config["model"] not in self.register["model"]:
            raise ValueError(f"Model {public_config['model']} not registered.")

        # Validate trainer
        if public_config["trainer"] not in self.register["trainer"]:
            raise ValueError(f"Trainer {public_config['trainer']} not registered.")

        return True
