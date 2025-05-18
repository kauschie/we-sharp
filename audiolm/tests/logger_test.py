import logging
import os

# Configure logging
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
print(f"Log directory: {log_dir}, Exists: {os.path.exists(log_dir)}")
log_file_path = os.path.join(log_dir, 'semantic_training.log')
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
logger.propagate = False






logging.info("printing here")