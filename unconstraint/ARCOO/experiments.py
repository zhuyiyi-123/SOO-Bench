
import os
import fire
from core.utils import setup_seed
from core.exp_conductor import ExpConductor


def dkitty_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="dkitty", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def ant_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="ant", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def supercond_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="supercond", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def tfbind8_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="tfbind8", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def utr_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="utr", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def chembl_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="chembl", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()
    

def hopper_arcoo_exp(seed):
    setup_seed(seed)
    ec = ExpConductor(task="hopper", algo="arcoo", label=("seed" + str(seed)))
    ec.config['random_seed'] = seed
    ec.run()

def hopper_arcoo_exp_hyper_m_emax(seed):
    for (init_m, k) in [(0.01, 32), (0.01, 128), (0.02, 64), (0.04, 32), (0.04, 128)]:
        setup_seed(seed)
        ec = ExpConductor(task="hopper", algo="arcoo", label=("seed" + str(seed)))
        ec.config['random_seed'] = seed
        ec.config['init_m'] = init_m
        ec.config['Ld_K_max'] = k
        ec.run()

def hopper_arcoo_exp_alation(seed):
    for (opt, train) in [(True, True), (False, True), (False, False)]:
        setup_seed(seed)
        ec = ExpConductor(task="hopper", algo="arcoo", label=("seed" + str(seed)))
        ec.config['e_train'] = train
        ec.config['opt_config']['energy_opt'] = opt
        ec.run()

def hopper_arcoo_exp_hyper_k(seed):
    for k in [2,4,8,16,32,64,128]:
        setup_seed(seed)
        ec = ExpConductor(task="hopper", algo="arcoo", label=("seed" + str(seed)))
        ec.config['random_seed'] = seed
        ec.config['Ld_K'] = k
        ec.run()

if __name__ == "__main__":
    fire.Fire()