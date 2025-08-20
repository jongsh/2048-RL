import os
import sys
import logging

from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs", log_name=None, level=logging.INFO):
        """
        Args:
            log_dir (str): Directory to save log files.
            log_name (str): Name of the log file. If None, uses current timestamp.
            level (int): Logging level, e.g., logging.INFO, logging.DEBUG.
        """
        if log_name is None:
            log_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{log_name}.log")
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # 避免重复打印

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(filename)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler for logging to a file
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def get_log_file(self):
        return self.log_path


if __name__ == "__main__":
    logger = Logger()
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    print(f"Logs are saved to: {logger.get_log_file()}")
