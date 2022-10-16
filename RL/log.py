# -*- coding: utf-8 -*-
# @File    : log.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import logging
import time
import os
from pathlib import Path


class MyLog:
    """
        日志记录
        mylog = Mylog()
        logger = mylog.logger
        logger.info()
        logger.warning()
    """
    def __init__(self, path: Path, filesave=False, consoleprint=True, name=None):
        """
        log
        :param path: 运行日志的当前文件 Path(__file__)
        :param filesave: 是否存储日志
        :param consoleprint: 是否打印到终端
        """

        self.formatter = logging.Formatter(
            f"{name}: %(message)s")
        self.logger = logging.getLogger(name=__name__)
        self.set_log_level()
        self.log_path = Path.joinpath(path.parent, 'Logs')
        if name is None:
            self.log_file = os.path.join(self.log_path, path.stem + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        else:
            self.log_file = os.path.join(self.log_path, path.stem + name)

            self.mk_log_dir()
        # log_path = os.path.join(os.getcwd(), 'Logs')

        if filesave:
            self.file_handler()

        if consoleprint:
            self.console_handler()

    def set_log_level(self, level=logging.DEBUG):
        self.logger.setLevel(level)

    def mk_log_dir(self):
        try:
            # os.mkdir(log_path)
            Path.mkdir(self.log_path)
        except FileExistsError:
            for child in self.log_path.iterdir():
                if child.stat().st_size == 0:
                    Path.unlink(child)

    def file_handler(self):
        fh = logging.FileHandler(self.log_file + '.log',
                                 mode='w', encoding='utf-8', )
        fh.setLevel(logging.INFO)

        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    def console_handler(self):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(self.formatter)

        self.logger.addHandler(ch)

    def pd_to_csv(self, dataframe):
        dataframe.to_csv(self.log_file + '.csv')
        self.logger.info("csv saved")
