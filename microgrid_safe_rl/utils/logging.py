import logging

def setup_logging(level='INFO'):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
