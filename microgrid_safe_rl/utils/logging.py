import logging
import sys

def setup_logging(level="INFO", debug_modules=None, quiet_modules=None):
    """
    Set up logging configuration with granular control.
    
    Args:
        level: Default logging level (INFO, WARNING, ERROR, DEBUG)
        debug_modules: List of module names to set to DEBUG level
        quiet_modules: List of module names to suppress (set to WARNING or ERROR)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Default quiet modules (always suppress these)
    default_quiet = [
        'numba', 'numba.core', 'numba.core.ssa', 'numba.core.byteflow',
        'numba.core.interpreter', 'pandas.core', 'matplotlib', 
        'urllib3', 'requests', 'pandapower.toolbox'
    ]
    
    if quiet_modules:
        default_quiet.extend(quiet_modules)
    
    # Apply quiet settings
    for module in default_quiet:
        logging.getLogger(module).setLevel(logging.ERROR)
    
    # Apply debug settings for specific modules
    if debug_modules:
        for module in debug_modules:
            logging.getLogger(module).setLevel(logging.DEBUG)
    
    # Return the root logger for convenience
    return root_logger