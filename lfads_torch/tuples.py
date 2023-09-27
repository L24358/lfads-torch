from collections import namedtuple 
from dataclasses import dataclass

SessionBatch = namedtuple(
    "SessionBatch",
    [
        "encod_data",
        "recon_data",
        "ext_input",
        "truth",
        "sv_mask",
    ],
)

SessionOutput = namedtuple(
    "SessionOutput",
    [
        "output_params",
        "factors",
        "ic_mean",
        "ic_std",
        "co_means",
        "co_stds",
        "gen_states",
        "gen_init",
        "gen_inputs",
        "con_states",
    ],
)

@dataclass
class SaveICAndInput:
    con_init = None
    gen_init = None
    factor_init = None
    ci = None