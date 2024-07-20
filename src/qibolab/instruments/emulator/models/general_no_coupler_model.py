from typing import Optional, Union, List

import numpy as np

from qibolab.instruments.emulator.models.methods import (
    default_noflux_platform_to_simulator_channels,
)

GHZ = 1e9

# model template for 0-1 system
def generate_default_params():
    # all time in ns and frequency in GHz
    """Returns template model parameters dictionary."""
    model_params = {
        "model_name": "general_no_coupler_model",
        "topology": [[0, 1]],
        "nqubits": 2,
        "ncouplers": 0,
        "qubits_list": ["0", "1"],
        "couplers_list": [],
        "nlevels_q": [2, 2],
        "nlevels_c": [],
        "readout_error": {
            # same key datatype as per runcard
            0: [0.01, 0.02],
            1: [0.01, 0.02],
        },
        "drive_freq": {
            "0": 5.0,
            "1": 5.1,
        },
        "T1": {
            "0": 0.0,
            "1": 0.0,
        },
        "T2": {
            "0": 0.0,
            "1": 0.0,
        },
        "lo_freq": {
            "0": 5.0,
            "1": 5.1,
        },
        "rabi_freq": {
            "0": 0.2,
            "1": 0.2,
        },
        "anharmonicity": {
            "0": -0.20,
            "1": -0.21,
        },
        "coupling_strength": {
            "1_0": 5.0e-3,
        },
    }
    return model_params


def get_model_params(
    platform_data_dict:dict,
    nlevels_q: Union[int, List[int]],
) -> dict:
    """Generates the model paramters for the general no coupler model.

    Args:
        platform_data_dict(dict): Dictionary containing the device data extracted from a device platform.
        nlevels_q(int, list): Number of levels for each qubit. If int, the same value gets assigned to all qubits.
        
    Returns:
        dict: Model parameters dictionary with all frequencies in GHz and times in ns that is required as an input to emulator runcards.

    Raises:
        ValueError: If length of nlevels_q does not match number of qubits when nlevels_q is a list.
    """
    model_params_dict = {'model_name': 'general_no_coupler_model'}
    model_params_dict |= {'topology': platform_data_dict['topology']}
    qubits_list = platform_data_dict['qubits_list']
    couplers_list = [] #platform_data_dict['couplers_list']
    characterization_dict = platform_data_dict['characterization']
    qubit_characterization_dict = characterization_dict['qubits']
    
    model_params_dict |= {'nqubits': len(qubits_list)}
    model_params_dict |= {'ncouplers': len(couplers_list)}
    model_params_dict |= {'qubits_list': [str(q) for q in qubits_list]}
    model_params_dict |= {'couplers_list': [str(c) for c in couplers_list]}

    if type(nlevels_q) == int:
        model_params_dict |= {'nlevels_q': [nlevels_q for q in qubits_list]}
    elif type(nlevels_q) == list:
        if len(nlevels_q)==len(qubits_list):
            model_params_dict |= {'nlevels_q': nlevels_q}
        else:
            raise ValueError(
                "Length of nlevels_q does not match number of qubits", len(qubits_list)
            )
    model_params_dict |= {'nlevels_c': []}

    drive_freq_dict = {}
    T1_dict = {}
    T2_dict = {}
    lo_freq_dict = {}
    rabi_freq_dict = {}
    anharmonicity_dict = {}
    readout_error_dict = {}

    for q in qubits_list:
        af = qubit_characterization_dict[q]['assignment_fidelity']
        if af == 0:
            readout_error_dict |= {str(q): [0.0, 0.0]}
        else:
            if type(af) is float:
                p0m1 = p1m0 = 1 - af
            else:
                p0m1, p1m0 = 1 - np.array(af)
            readout_error_dict |= {str(q): [p0m1,p1m0]}
        drive_freq_dict |= {str(q): qubit_characterization_dict[q]['drive_frequency']/GHZ}
        T1_dict |= {str(q): qubit_characterization_dict[q]['T1']}
        T2_dict |= {str(q): qubit_characterization_dict[q]['T2']}
        lo_freq_dict |= {str(q): qubit_characterization_dict[q]['drive_frequency']/GHZ}
        rabi_freq_dict |= {str(q): qubit_characterization_dict[q]['rabi_frequency']/GHZ}
        anharmonicity_dict |= {str(q): qubit_characterization_dict[q]['anharmonicity']/GHZ}

    model_params_dict |= {'readout_error': readout_error_dict}
    model_params_dict |= {'drive_freq': drive_freq_dict}
    model_params_dict |= {'T1': T1_dict}
    model_params_dict |= {'T2': T2_dict}
    model_params_dict |= {'lo_freq': lo_freq_dict}
    model_params_dict |= {'rabi_freq': rabi_freq_dict}
    model_params_dict |= {'anharmonicity': anharmonicity_dict}    
    model_params_dict |= {'coupling_strength': {}}
    
    return model_params_dict


def generate_model_config(
    model_params: dict = None,
    nlevels_q: Optional[list] = None,
    topology: Optional[list] = None,
) -> dict:
    """Generates the model configuration dictionary.

    Args:
        model_params(dict): Dictionary containing the model parameters.
        nlevels_q(list, optional): List of the dimensions of each qubit to be simulated, in big endian order. Defaults to None, in which case it will use the values of model_params['nlevels_q'] will be used.
        topology(list, optional): List containing all pairs of qubit indices that are nearest neighbours. Defaults to none, in which case the value of model_params['topology'] will be used.

    Returns:
        dict: Model configuration dictionary with all frequencies in GHz and times in ns.
    """
    if model_params is None:
        model_params = generate_default_params()

    # allows for user to overwrite topology in model_params for quick test
    if topology is None:
        topology = model_params["topology"]

    model_name = model_params["model_name"]
    readout_error = model_params["readout_error"]
    qubits_list = model_params["qubits_list"]

    rabi_freq_dict = model_params["rabi_freq"]

    if nlevels_q is None:
        nlevels_q = model_params["nlevels_q"]

    drift_hamiltonian_dict = {"one_body": [], "two_body": []}
    drive_hamiltonian_dict = {}

    dissipation_dict = {"t1": [], "t2": []}

    # generate instructions
    # single qubit terms
    for i, q in enumerate(qubits_list):
        # drift Hamiltonian terms (constant in time)
        drift_hamiltonian_dict["one_body"].append(
            (2 * np.pi * model_params["lo_freq"][q], f"O_{q}", [q])
        )
        drift_hamiltonian_dict["one_body"].append(
            (
                np.pi * model_params["anharmonicity"][q],
                f"O_{q} * O_{q} - O_{q}",
                [q],
            )
        )

        # drive Hamiltonian terms (amplitude determined by pulse sequence)
        drive_hamiltonian_dict.update({f"D-{qubits_list[i]}": []})
        drive_hamiltonian_dict[f"D-{qubits_list[i]}"].append(
            (2 * np.pi * model_params["rabi_freq"][q], f"X_{q}", [q])
        )

        # dissipation terms (one qubit, constant in time)
        t1 = model_params["T1"][q]
        g1 = 0 if t1 == 0 else 1.0 / t1 * 2 * np.pi
        t2 = model_params["T2"][q]
        g2 = 0 if t1 == 0 else 1.0 / t2 * 2 * np.pi

        dissipation_dict["t1"].append((np.sqrt(g1 / 2), f"sp01_{q}", [q]))
        dissipation_dict["t2"].append((np.sqrt(g2 / 2), f"Z01_{q}", [q]))

    # two-body terms (couplings)
    for key in list(model_params["coupling_strength"].keys()):
        ind2, ind1 = key.split(
            "_"
        )  # ind2 > ind1 with ind_qubit > ind_coupler as per Hilbert space ordering
        coupling = model_params["coupling_strength"][key]
        drift_hamiltonian_dict["two_body"].append(
            (
                2 * np.pi * coupling,
                f"bdag_{ind2} ^ b_{ind1} + b_{ind2} ^ bdag_{ind1}",
                [ind2, ind1],
            )
        )

    model_config = {
        "model_name": model_name,
        "topology": topology,
        "qubits_list": qubits_list,
        "nlevels_q": nlevels_q,
        "couplers_list": [],
        "nlevels_c": [],
        "drift": drift_hamiltonian_dict,
        "drive": drive_hamiltonian_dict,
        "dissipation": dissipation_dict,
        "method": "master_equation",
        "readout_error": readout_error,
        "platform_to_simulator_channels": default_noflux_platform_to_simulator_channels(
            qubits_list, couplers_list=[]
        ),
    }

    return model_config
