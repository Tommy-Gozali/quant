from yaml import safe_load

config = safe_load(open("config\config.yaml"))
TICKER      = config["data"]["TICKER"]
N_IN        = config["data"]["N_IN"]
N_OUT       = config["data"]["N_OUT"]
CASH        = config["trade"]["CASH"]
COMMISSION  = config["trade"]["COMMISSION"]

FAST        = 20
SLOW        = 50

N_PERMS     = 1000       # permutation test iterations
RF_DAILY    = 0.02 / 252