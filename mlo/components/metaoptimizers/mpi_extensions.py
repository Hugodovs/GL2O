import os
import pickle
import logging

import numpy as np

class MPISaver:

    def __init__(self, file_path):

        self.file_path = file_path
        self.folder_path, self.filename = os.path.split(self.file_path)

        try:
            os.makedirs(self.folder_path)
        except:
            pass

        #assert os.path.isfile(self.file_path) == False, f'You should delete all files inside the folder: {self.folder_path} | {self.file_path}'

    def write(self, data):
        with open(self.file_path, 'ab') as f:
            np.save(f, data)

class MPILogger:

    def __init__(self, file_path):

        self.file_path = file_path
        self.folder_path, self.filename = os.path.split(file_path)

        try:
            os.makedirs(self.folder_path)
        except:
            pass

        #assert os.path.isfile(self.file_path) == False, f'You should delete all files inside the folder: {self.folder_path} | {self.file_path}'

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.file_handler = logging.FileHandler(self.file_path, 'w')
        self.logger.addHandler(self.file_handler)
         

    def debug(self, msg):
        self.logger.debug(msg)
        exit()