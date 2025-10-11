import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[97m',    # White
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m'  # Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_level = record.levelname
        log_color = self.COLORS.get(log_level, self.RESET)
        log_message = super().format(record)
        return f"{log_color}{log_message}{self.RESET}"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    colored_formatter = ColoredFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    for handler in root_logger.handlers:
        handler.setFormatter(colored_formatter)
