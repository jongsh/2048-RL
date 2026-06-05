from omegaconf import OmegaConf, DictConfig, ListConfig


class Configuration:
    def __init__(self, config_path="configs/config.yaml", cli_args=None, from_scratch=True):
        self.config = None
        # Load a pre-saved configuration
        if not from_scratch:
            self.config = OmegaConf.load(config_path)

        # Load configurations from multiple sources
        if not self.config or any(key not in self.config for key in ["env", "agent", "model", "trainer"]):
            self.config = OmegaConf.load(config_path)
            self.config["env"] = OmegaConf.load(self.config["paths"]["env"])
            self.config["agent"] = OmegaConf.load(self.config["paths"]["agent"])
            self.config["model"] = OmegaConf.load(self.config["paths"]["model"])
            self.config["trainer"] = OmegaConf.load(self.config["paths"]["trainer"])

        # Override config with command-line arguments
        if cli_args:
            dotlist = [arg[2:] for arg in cli_args if arg.startswith("--")]
            cli_conf = OmegaConf.from_dotlist(dotlist)
            self.config = OmegaConf.merge(self.config, cli_conf)

        if self._validate():
            print("Configuration is valid.")

    def _validate(self):
        """validate the configuration structure"""
        public_section = None
        for key in ["env", "agent", "model", "trainer"]:
            if public_section is None:
                public_section = self.config[key]["public"]
            else:
                for k, v in public_section.items():
                    if k not in self.config[key]["public"] or self.config[key]["public"][k] != v:
                        raise ValueError(
                            f"All public configurations must be consistent across sections, but found inconsistency in key: {k}"
                        )
        return True

    def __getitem__(self, key):
        """Get a specific section of the configuration"""
        return self.config.get(key, None)

    def save_config(self, dir_path):
        """Save the configuration to the specified directory"""
        save_path = f"{dir_path}/config.yaml"
        OmegaConf.save(self.config, save_path)

    def to_string(self):
        """Return the configuration as a formatted string"""

        def format_value(v, indent=2):
            if isinstance(v, DictConfig) or isinstance(v, dict):
                tmp_lines = [""]
                for k2, v2 in v.items():
                    sub_str = format_value(v2, indent + 2)
                    if k2 in ["env", "agent", "model", "trainer"] and indent == 0:
                        tmp_lines.append("\n" + " " * indent + f"[{k2}]{sub_str}")
                    else:
                        tmp_lines.append(" " * indent + f"{k2}: {sub_str}")
                return "\n".join(tmp_lines)

            elif isinstance(v, ListConfig) or isinstance(v, list):
                return "[" + ", ".join(map(str, v)) + "]"
            else:
                return str(v)

        config_str = "=" * 10 + " Configuration Summary " + "=" * 10
        config_str += format_value(self.config, indent=0)
        config_str += "\n" + "=" * 40
        return config_str
