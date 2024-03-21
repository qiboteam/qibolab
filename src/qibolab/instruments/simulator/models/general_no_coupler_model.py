from typing import Union

import numpy as np

from qibolab.instruments.simulator.models.methods import (
    GHz,
    default_noflux_platform2simulator_channels,
    load_model_params,
)


def generate_model_config(
    model_params: Union[dict, str], nlevels_q: list = None, topology: list = None
) -> dict:
    """Generates the model configuration dictionary.

    Args:
        model_params(dict or str): Dictionary with model paramters or path of a yaml file (string) containing the model parameters.
        nlevels_q(list, optional): List of the dimensions of each qubit to be simulated, in big endian order. Defaults to none, in which case a list of 2s with the same length as model_params['qubits_list'] will be used.
        topology(list, optional): List containing all pairs of qubit indices that are nearest neighbours. Defaults to none, in which case the value of model_params['topology'] will be used.

    Returns:
        dict: Model configuration dictionary with all frequencies in GHz and times in ns.
    """
    model_params = load_model_params(model_params)

    # allows for user to overwrite topology in model_params for quick test
    if topology is None:
        topology = model_params["topology"]

    device_name = model_params["device_name"]
    sampling_rate = model_params["sampling_rate"] / GHz  # units of samples/ns
    readout_error = model_params["readout_error"]
    # nqubits = model_params['nqubits']
    ncouplers = model_params["ncouplers"]
    # runcard_qubits_list = model_params['qubits_list']
    qubits_list = model_params["qubits_list"]

    rabi_freq_dict = model_params["rabi_freq"]

    if nlevels_q == None:
        nlevels_q = [2 for q in qubits_list]

    drift_hamiltonian_dict = {"one_body": [], "two_body": []}
    drive_hamiltonian_dict = {}

    dissipation_dict = {"t1": [], "t2": []}

    # generate instructions
    # single qubit terms
    for i, q in enumerate(qubits_list):
        # drift Hamiltonian terms (constant in time)
        drift_hamiltonian_dict["one_body"].append(
            (2 * np.pi * model_params["lo_freq"][q] / GHz, f"O_{q}", [q])
        )
        drift_hamiltonian_dict["one_body"].append(
            (
                np.pi * model_params["anharmonicity"][q] / GHz,
                f"O_{q} * O_{q} - O_{q}",
                [q],
            )
        )

        # drive Hamiltonian terms (amplitude determined by pulse sequence)
        drive_hamiltonian_dict.update({f"D-{qubits_list[i]}": []})
        drive_hamiltonian_dict[f"D-{qubits_list[i]}"].append(
            (2 * np.pi * model_params["rabi_freq"][q] / GHz, f"X_{q}", [q])
        )

        # dissipation terms (one qubit, constant in time)
        t1 = model_params["T1"][q]
        g1 = 0 if t1 == 0 else 1.0 / t1 * 2 * np.pi / GHz
        t2 = model_params["T2"][q]
        g2 = 0 if t1 == 0 else 1.0 / t2 * 2 * np.pi / GHz

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
                2 * np.pi * coupling / GHz,
                f"bdag_{ind2} ^ b_{ind1} + b_{ind2} ^ bdag_{ind1}",
                [ind2, ind1],
            )
        )

        # couplers_list = []
        # nlevels_c = []

    model_config = {
        "model_name": "general no coupler CR drive model",
        "device_name": device_name,
        "sampling_rate": sampling_rate,
        "runcard_duration_in_dt_units": False,
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
        "platform2simulator_channels": default_noflux_platform2simulator_channels(
            qubits_list, couplers_list=[]
        ),
    }

    return model_config


def generate_model_config_oneQ(model_params: dict, nlevels_q: int = None) -> dict:
    """Calls `generate_model_config` to construct a model configuration
    dictionary for generic 1Q emulators."""
    model_config = generate_model_config(
        model_params, nlevels_q=[nlevels_q], topology=[]
    )

    return model_config
