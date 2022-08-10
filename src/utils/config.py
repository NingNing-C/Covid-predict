import yaml
from easydict import EasyDict
import os

def get_config(args, logger=None):
    # cfg_path = os.path.join(args.experiment_path, 'config.yaml')
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        # print_log("Failed to resume", logger = logger)
        raise FileNotFoundError()
    # print_log(f'Resume yaml from {cfg_path}', logger = logger)
    args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config