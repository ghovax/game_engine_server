import logging


class ColorCodes:
    GREY = "\x1b[38;21m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"


class ColoredFormatter(logging.Formatter):
    def __init__(self, format_string):
        super().__init__(format_string)
        self.FORMATS = {
            logging.DEBUG: ColorCodes.GREY + format_string + ColorCodes.RESET,
            logging.INFO: ColorCodes.BLUE + format_string + ColorCodes.RESET,
            logging.WARNING: ColorCodes.YELLOW + format_string + ColorCodes.RESET,
            logging.ERROR: ColorCodes.RED + format_string + ColorCodes.RESET,
            logging.CRITICAL: ColorCodes.BOLD_RED + format_string + ColorCodes.RESET,
        }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


def setup_logging():
    format_str = "%(levelname)s:%(name)s:%(message)s"
    colored_formatter = ColoredFormatter(format_str)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(colored_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

    # Configure Flask's logger
    flask_logger = logging.getLogger("werkzeug")
    flask_logger.setLevel(logging.INFO)

    # Configure PyAssimp logger
    pyassimp_logger = logging.getLogger("pyassimp")
    pyassimp_logger.setLevel(logging.INFO)
