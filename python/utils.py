import logging
import os
import sys
from datetime import date
from logging.handlers import TimedRotatingFileHandler

import pandas as pd

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_DIR = "./logs/"
LOG_FILE = "my_app.log" + date.today().strftime("%m%d%Y")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    file_handler = TimedRotatingFileHandler(os.path.join(LOG_DIR, LOG_FILE), when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def loadDataFromCsv(filePath: str,
                    rows: int = -1):  # [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    print("reading csv...")
    if rows > 0:
        df = pd.read_csv(filePath, nrows=rows)
    else:
        df = pd.read_csv(filePath)

    return df