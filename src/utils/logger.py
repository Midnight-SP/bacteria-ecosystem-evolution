import logging
import csv

def setup_logger(name="ecosystem", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

class StatsLogger:
    def __init__(self, path):
        self.path = path
        self.header_written = False

    def log(self, data: dict):
        write_header = not self.header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(data)