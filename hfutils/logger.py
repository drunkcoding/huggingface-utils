import logging
import os
import sys
from logging import handlers
from datetime import datetime
import uuid

class Logger():

    def __init__(self, file, level, rollsize, backup, mode="a"):        
        logging.addLevelName(5, "TRACE")

        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        format = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(process)d [%(filename)s - %(lineno)d] -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S.%f",
        )

        filename = file + ".log"

        self.level = level
        self.fp = open(filename, mode)

    def __del__(self):
        self.fp.close()

    def info(self, pattern, *args):
        if self.level > logging.INFO: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "INFO", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def fatal(self, pattern, *args):
        if self.level > logging.FATAL: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "FATAL", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def critical(self, pattern, *args):
        if self.level > logging.CRITICAL: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "CRITICAL", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def debug(self, pattern, *args):
        if self.level > logging.DEBUG: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "DEBUG", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def error(self, pattern, *args):
        if self.level > logging.ERROR: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "ERROR", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def warn(self, pattern, *args):
        if self.level > logging.WARN: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "WARN", os.getpid()) + (pattern % args), file=self.fp, flush=True)

    def trace(self, pattern, *args):
        if self.level > 5: return
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("%s %s %s  " % (date_time, "TRACE", os.getpid()) + (pattern % args), file=self.fp, flush=True)

if __name__ == "__main__":
    logger = Logger(__file__, logging.INFO, 0, 0)
    uid = uuid.uuid4().hex
    namespace = "test"
    start_time = 0
    end_time = 1
    start_power = 2
    end_power = 3
    logger.info("[%s,%s] %s inference %s %s",
            uid, namespace,
            (start_time, end_time, start_power, end_power),
            end_time - start_time)

# class Logger():

#     def __init__(self, file, level, rollsize, backup, stdout=False):        
#         # if level.lower() == "debug":
#         #     self.level = logging.DEBUG
#         # elif level.lower() == "info":
#         #     self.level = logging.INFO
#         # else:
#         #     raise ValueError('logging level must be INFO or DEBUG')

#         logging.addLevelName(5, "TRACE")

#         self.logger = logging.getLogger()
#         self.logger.setLevel(level)

#         format = logging.Formatter(
#             fmt="%(asctime)s - %(levelname)s - %(process)d [%(filename)s - %(lineno)d] -   %(message)s",
#             datefmt="%m/%d/%Y %H:%M:%S.%f",
#         )

#         if stdout:
#             ch = logging.StreamHandler(sys.stdout)
#             ch.setFormatter(format)
#             self.logger.addHandler(ch)

#         filename = file.split(".")[0]

#         fh = handlers.RotatingFileHandler(f"{filename}.log", maxBytes=rollsize, backupCount=backup)
#         fh.setFormatter(format)
#         self.logger.addHandler(fh)

#     def info(self, pattern, *args):
#         self.logger.info(pattern, *args)

#     def fatal(self, pattern, *args):
#         self.logger.fatal(pattern, *args)

#     def critical(self, pattern, *args):
#         self.logger.critical(pattern, *args)

#     def debug(self, pattern, *args):
#         self.logger.debug(pattern, *args)

#     def error(self, pattern, *args):
#         self.logger.error(pattern, *args)

#     def warn(self, pattern, *args):
#         self.logger.warn(pattern, *args)

#     def trace(self, pattern, *args):
#         self.logger.log(5, pattern, *args)
 
# if __name__ == "__main__":
#     logger = Logger(__file__, "info", 0, 0)
#     logger.info("abcdefg")

