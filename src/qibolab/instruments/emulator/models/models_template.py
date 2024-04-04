import numpy as np

from qibolab.instruments.emulator.models.methods import (
    default_noflux_platform2simulator_channels,
)


def generate_default_params():
    # all time in ns and frequency in GHz
    """Returns template model parameters dictionary."""
    model_params = {
        "device_name": "model template for 0-c1-1 system",
        "topology": [[0, 1]],
        "nqubits": 2,
        "ncouplers": 1,
        "qubits_list": ["0", "1"],
        "couplers_list": ["c1"],
        "sampling_rate": 2.0,  # units of samples/ns
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
            "c1": 6.5,
        },
        "rabi_freq": {
            "0": 0.2,
            "1": 0.2,
        },
        "anharmonicity": {
            "0": -0.20,
            "1": -0.21,
            "c1": -0.1,
        },
        "coupling_strength": {
            "1_c1": 101.0e-3,
            "0_c1": 100.0e-3,
            "1_0": 5.0e-3,
        },
    }
    return model_params


def generate_model_config(
    model_params: dict = None,
    nlevels_q: list = None,
    nlevels_c: list = None,
    topology: list = None,
) -> dict:
    """Generates a template model configuration dictionary.

    Args:
        model_params(dict): Dictionary containing the model parameters.
        nlevels_q(list, optional): List of the dimensions of each qubit to be simulated, in big endian order. Defaults to none, in which case a list of 2s with the same length as model_params['qubits_list'] will be used.
        nlevels_c(list, optional): List of the dimensions of each coupler to be simulated, in big endian order. Defaults to none, in which case a list of 2s with the same length as model_params['couplers_list'] will be used.
        topology(list, optional): List containing all pairs of qubit indices that are nearest neighbours. Defaults to none, in which case the value of model_params['topology'] will be used.

    Returns:
        dict: Model configuration dictionary with all frequencies in GHz and times in ns.
    """
    if model_params is None:
        model_params = generate_default_params()

    # allows for user to overwrite topology in model_params for quick test
    if topology is None:
        topology = model_params["topology"]

    device_name = model_params["device_name"]
    sampling_rate = model_params["sampling_rate"]
    readout_error = model_params["readout_error"]
    qubits_list = model_params["qubits_list"]
    couplers_list = model_params["couplers_list"]

    if nlevels_q is None:
        nlevels_q = [2 for q in qubits_list]
    if nlevels_c is None:
        nlevels_c = [2 for c in couplers_list]

    rabi_freq_dict = model_params["rabi_freq"]

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
        g1 = 0 if t1 == 0 else 1.0 / t1
        t2 = model_params["T2"][q]
        g2 = 0 if t1 == 0 else 1.0 / t2

        dissipation_dict["t1"].append((np.sqrt(g1), f"sp01_{q}", [q]))
        dissipation_dict["t2"].append((np.sqrt(g2), f"Z01_{q}", [q]))

    # single coupler terms
    for i, c in enumerate(couplers_list):
        # drift Hamiltonian terms (constant in time)
        drift_hamiltonian_dict["one_body"].append(
            (2 * np.pi * model_params["lo_freq"][c], f"O_{c}", [c])
        )
        drift_hamiltonian_dict["one_body"].append(
            (
                np.pi * model_params["anharmonicity"][c],
                f"O_{c} * O_{c} - O_{c}",
                [c],
            )
        )

    ## two-body terms (couplings)
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
        "model_name": "template for general (no flux) model",
        "device_name": device_name,
        "sampling_rate": sampling_rate,
        "runcard_duration_in_dt_units": False,
        "topology": topology,
        "qubits_list": qubits_list,
        "nlevels_q": nlevels_q,
        "couplers_list": couplers_list,
        "nlevels_c": nlevels_c,
        "drift": drift_hamiltonian_dict,
        "drive": drive_hamiltonian_dict,
        "dissipation": dissipation_dict,
        "method": "master_equation",
        "readout_error": readout_error,
        "platform2simulator_channels": default_noflux_platform2simulator_channels(
            qubits_list, couplers_list
        ),
    }

    return model_config
