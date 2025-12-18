import logging
logging.basicConfig(level=logging.INFO)

from accelerate import Accelerator
accelerator = Accelerator()

logging.info(f"Using accelerator: {accelerator}")
logging.info(f"Config: {accelerator.state}")
logging.info(f"Device: {accelerator.device}")
logging.info(f"Mixed precision: {accelerator.mixed_precision}")