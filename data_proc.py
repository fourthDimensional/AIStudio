import os
import utils
import logging

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)


class Preprocessor_Layer:
    def create_layer(self):
        raise NotImplementedError


