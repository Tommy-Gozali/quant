from yaml import safe_load

config = safe_load(open("config.yaml"))
TICKER      = config["data"]["TICKER"]
N_IN        = config["data"]["N_IN"]
N_OUT       = config["data"]["N_OUT"]
CASH        = config["trade"]["CASH"]
COMMISSION  = config["trade"]["COMMISSION"]