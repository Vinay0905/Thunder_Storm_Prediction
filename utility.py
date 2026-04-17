import logging 
import yaml
import os 

#Why are we doing this?
#Logging: 
#   In production, you won't be looking at Jupyter cell outputs. 
#   You'll check log files to see if the data merged correctly.
#Config: 
#   It makes your project portable. 
#   If you move this to a server, you only change the YAML.

def setup_logging(log_file="pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path="config.yaml"):
    with open(config_path,"r") as f:
        return yaml.safe_load(f)
def ensure_dir(path):
    os.makedirs(os.path.dirname(path),exist_ok=True)


