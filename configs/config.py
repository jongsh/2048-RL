from omegaconf import OmegaConf


class Configuration:
    def __init__(self, config_path="configs/config.yaml", cli_args=None, from_scratch=True):
        if from_scratch:  # Load configurations from multiple sources
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

        else:  # Load a pre-saved configuration
            self.config = OmegaConf.load(config_path)

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
        for section, content in self.config.items():
            lines.append(f"[{section}]")
            lines.append(format_value(content, indent=2))
            lines.append("")
        lines.append("=" * 40)
        return "\n".join(lines)
