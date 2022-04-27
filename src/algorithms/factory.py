from algorithms.curl import CURL
from algorithms.drq import DrQ
from algorithms.pad import PAD
from algorithms.rad import RAD
from algorithms.sac import SAC
from algorithms.sac_bc import SAC_BC
from algorithms.sac_exp import SAC_EXP
from algorithms.sac_feat import SAC_FEAT
from algorithms.sac_feat_exp import SAC_FEAT_EXP
from algorithms.sac_nsfp import SAC_NSFP
from algorithms.sac_offline_exp import SAC_OFFLINE_EXP
from algorithms.sac_rev import SAC_REV
from algorithms.soda import SODA
from algorithms.svea import SVEA

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "sac_exp": SAC_EXP,
    "sac_feat": SAC_FEAT,
    "sac_feat_exp": SAC_FEAT_EXP,
    "sac_bc": SAC_BC,
    "sac_rev": SAC_REV,
    "sac_nsfp": SAC_NSFP,
    "sac_offline_exp": SAC_OFFLINE_EXP,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
