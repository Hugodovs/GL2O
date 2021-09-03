import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG)

x = 10
logging.debug(f'Add:{x}')