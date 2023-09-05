import sys
import datetime


def log_function_settlement() -> None:
    now = datetime.datetime.now()
    log_save_txt = "./log_file/log_file_" + str(now)[:19] + ".txt"  # linux: save log file in log_file folder
    save_terminal_logging(log_save_txt)


def save_terminal_logging(log_save_path: str) -> None:
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    sys.stdout = Logger(log_save_path)