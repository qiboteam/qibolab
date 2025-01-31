from typing import List, Optional, Union

import numpy as np

from qibolab.instruments.emulator.models.methods import (
    default_platform_to_simulator_channels,
    default_platform_to_simulator_qubits,
)

GHZ = 1e9


# model template for 0-1 system
def generate_default_params():
    # all time in ns and frequency in GHz
    """Returns template model parameters dictionary."""
    model_params = {
        "model_name": "general_coupler_model",
        "topology": [[0, 1]],
        "nqubits": 2,
        "ncouplers": 1,
        "qubits_list": ["0", "1"],
        "couplers_list": ["c0"],
        "nlevels_q": [2, 2],
        "nlevels_c": [2],
        "readout_error": {
            # same key datatype as per runcard
            0: [0.01, 0.02],
            1: [0.01, 0.02],
        },
        "drive_freq": {
            "0": 5.0,
            "1": 5.1,
            "c0": 6.4,
        },
        "T1": {
            "0": 0.0,
            "1": 0.0,
            "c0": 0.0,
        },
        "T2": {
            "0": 0.0,
            "1": 0.0,
            "c0": 0.0,
        },
        "max_lo_freq": {
            "0": 5.0,
            "1": 5.1,
            "c0": 6.4,
        },
        "flux_quanta": {
            "0": 0.3183098861837907,
            "1": 0.3183098861837907,
            "c0": 0.3183098861837907,
        },
        "rabi_freq": {
            "0": 0.2,
            "1": 0.2,
            "c0": 0.0,
        },
        "anharmonicity": {
            "0": -0.20,
            "1": -0.21,
            "c0": 0.0,
        },
        "coupling_strength": {
            "1_0": -0.005,
            "1_c0": 0.1,
            "0_c0": 0.1,
        },
    }
    return model_params


# to add relabel_qubits
"""
emulator_qubits_list = []
    for i,q in enumerate(qubits_list):
        if relabel_qubits:
            i = str(i)
        else:
            i = q
        emulator_qubits_list.append(int(i))
        """


def get_model_params(
    platform_data_dict: dict,
    nlevels_q: Union[int, List[int]],
    nlevels_c: Union[int, List[int]],
    relabel_qubits: bool = False,
) -> dict:
    """Generates the model paramters for the general coupler model.

    Args:
        platform_data_dict(dict): Dictionary containing the device data extracted from a device platform.
        nlevels_q(int, list): Number of levels for each qubit. If int, the same value gets assigned to all qubits.
        nlevels_c(int, list): Number of levels for each coupler. If int, the same value gets assigned to all couplers.
        relabel_qubits(bool): Flag to relabel qubits to ascending integers. False by default.

    Returns:
        dict: Model parameters dictionary with all frequencies in GHz and times in ns that is required as an input to emulator runcards.

    Raises:
        ValueError: If length of nlevels_q does not match number of qubits when nlevels_q is a list.
    """
    model_params_dict = {"model_name": "general_coupler_model"}
    model_params_dict |= {"topology": platform_data_dict["topology"]}
    qubits_list = platform_data_dict["qubits_list"]
    couplers_list = platform_data_dict["couplers_list"]
    characterization_dict = platform_data_dict["characterization"]
    qubit_characterization_dict = characterization_dict["qubits"]

    drive_freq_dict = {}
    T1_dict = {}
    T2_dict = {}
    max_lo_freq_dict = {}
    flux_quanta_dict = {}
    rabi_freq_dict = {}
    anharmonicity_dict = {}
    readout_error_dict = {}

    relabelled_qubits_list = []
    for i, q in enumerate(qubits_list):
        if relabel_qubits:
            i = str(i)
        else:
            i = str(q)
        relabelled_qubits_list.append(i)

        af = qubit_characterization_dict[q]["assignment_fidelity"]
        if af == 0:
            readout_error_dict |= {i: [0.0, 0.0]}
        else:
            if type(af) is float:
                p0m1 = p1m0 = 1 - af
            else:
                p0m1, p1m0 = 1 - np.array(af)
            readout_error_dict |= {i: [p0m1, p1m0]}
        drive_freq_dict |= {i: qubit_characterization_dict[q]["drive_frequency"] / GHZ}
        T1_dict |= {i: qubit_characterization_dict[q]["T1"]}
        T2_dict |= {i: qubit_characterization_dict[q]["T2"]}
        max_lo_freq_dict |= {i: qubit_characterization_dict[q]["max_lo_freq"] / GHZ}
        rabi_freq_dict |= {i: qubit_characterization_dict[q]["rabi_frequency"] / GHZ}
        anharmonicity_dict |= {i: qubit_characterization_dict[q]["anharmonicity"] / GHZ}
        # flux_quanta_dict |= {i: 0.1} # to be updated

    """
    # TODO
    for c in couplers_list:
        drive_freq_dict |= {str(c): qubit_characterization_dict[c]['drive_frequency']/GHZ}
        T1_dict |= {str(c): qubit_characterization_dict[c]['T1']}
        T2_dict |= {str(c): qubit_characterization_dict[c]['T2']}
        max_lo_freq_dict |= {str(c): qubit_characterization_dict[c]['max_lo_freq']/GHZ}
        rabi_freq_dict |= {str(c): qubit_characterization_dict[c]['rabi_frequency']/GHZ}
        anharmonicity_dict |= {str(c): qubit_characterization_dict[c]['anharmonicity']/GHZ}
        #flux_quanta_dict |= {str(c): 0.1} # to be updated
    """

    model_params_dict |= {"nqubits": len(qubits_list)}
    model_params_dict |= {"ncouplers": len(couplers_list)}
    model_params_dict |= {"qubits_list": relabelled_qubits_list}
    model_params_dict |= {"couplers_list": [str(c) for c in couplers_list]}

    if type(nlevels_q) == int:
        model_params_dict |= {"nlevels_q": [nlevels_q for q in qubits_list]}
    elif type(nlevels_q) == list:
        if len(nlevels_q) == len(qubits_list):
            model_params_dict |= {"nlevels_q": nlevels_q}
        else:
            raise ValueError(
                "Length of nlevels_q does not match number of qubits", len(qubits_list)
            )
    if type(nlevels_c) == int:
        model_params_dict |= {"nlevels_c": [nlevels_c for c in couplers_list]}
    elif type(nlevels_c) == list:
        if len(nlevels_c) == len(couplers_list):
            model_params_dict |= {"nlevels_c": nlevels_c}
        else:
            raise ValueError(
                "Length of nlevels_c does not match number of couplers",
                len(couplers_list),
            )

    model_params_dict |= {"readout_error": readout_error_dict}
    model_params_dict |= {"drive_freq": drive_freq_dict}
    model_params_dict |= {"T1": T1_dict}
    model_params_dict |= {"T2": T2_dict}
    model_params_dict |= {"max_lo_freq": max_lo_freq_dict}
    model_params_dict |= {"flux_quanta": flux_quanta_dict}
    model_params_dict |= {"rabi_freq": rabi_freq_dict}
    model_params_dict |= {"anharmonicity": anharmonicity_dict}
    model_params_dict |= {"coupling_strength": {}}

    return model_params_dict


def generate_model_config(
    model_params: dict = None,
    nlevels_q: Optional[list] = None,
    nlevels_c: Optional[list] = None,
    topology: Optional[list] = None,
) -> dict:
    """Generates the model configuration dictionary.

    Args:
        model_params(dict): Dictionary containing the model parameters.
        nlevels_q(list, optional): List of the dimensions of each qubit to be simulated, in big endian order. Defaults to None, in which case it will use the values of model_params['nlevels_q'] will be used.
        nlevels_c(list, optional): List of the dimensions of each coupler to be simulated, in big endian order. Defaults to None, in which case it will use the values of model_params['nlevels_c'] will be used.
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
    # qubits_list = model_params["qubits_list"]
    device_qubits_list = model_params["qubits_list"]
    # couplers_list = model_params["couplers_list"]
    device_couplers_list = model_params["couplers_list"]

    platform_to_simulator_qubits = default_platform_to_simulator_qubits(
        device_qubits_list, device_couplers_list
    )
    qubits_list = [platform_to_simulator_qubits[q] for q in device_qubits_list]
    couplers_list = [platform_to_simulator_qubits[c] for c in device_couplers_list]

    if nlevels_q is None:
        nlevels_q = model_params["nlevels_q"]
    if nlevels_c is None:
        nlevels_c = model_params["nlevels_c"]

    drift_hamiltonian_dict = {"one_body": [], "two_body": []}
    drive_hamiltonian_dict = {}
    flux_hamiltonian_dict = {}
    flux_params_dict = {}

    dissipation_dict = {"t1": [], "t2": []}

    # generate instructions
    # single qubit terms
    # for i, q in enumerate(qubits_list):
    for q in device_qubits_list:
        ind = platform_to_simulator_qubits[q]
        # drift Hamiltonian terms (constant in time)
        drift_hamiltonian_dict["one_body"].append(
            # (2 * np.pi * model_params["drive_freq"][q], f"O_{q}", [q])
            (2 * np.pi * model_params["drive_freq"][q], f"O_{ind}", [ind])
        )
        drift_hamiltonian_dict["one_body"].append(
            (
                np.pi * model_params["anharmonicity"][q],
                # f"O_{q} * O_{q} - O_{q}",
                f"O_{ind} * O_{ind} - O_{ind}",
                # [q],
                [ind],
            )
        )

        # drive Hamiltonian terms (amplitude determined by pulse sequence)
        # drive_hamiltonian_dict.update({f"D-{qubits_list[i]}": []})
        drive_hamiltonian_dict.update({f"D-{ind}": []})
        # drive_hamiltonian_dict[f"D-{qubits_list[i]}"].append(
        #    (2 * np.pi * model_params["rabi_freq"][q], f"X_{q}", [q])
        # )
        drive_hamiltonian_dict[f"D-{ind}"].append(
            (2 * np.pi * model_params["rabi_freq"][q], f"X_{ind}", [ind])
        )

        # flux Hamiltonian terms (amplitude determined by processed pulse sequence)
        # flux_hamiltonian_dict.update({f"F-{qubits_list[i]}": []})
        flux_hamiltonian_dict.update({f"F-{ind}": []})
        # flux_hamiltonian_dict[f"F-{qubits_list[i]}"].append((2 * np.pi, f"O_{q}", [q]))
        flux_hamiltonian_dict[f"F-{ind}"].append((2 * np.pi, f"O_{ind}", [ind]))

        # flux detuning parameters:
        try:
            flux_params_dict |= {
                # q: {
                ind: {
                    "flux_quanta": model_params["flux_quanta"][q],
                    "max_frequency": model_params["max_lo_freq"][q],
                    "current_frequency": model_params["drive_freq"][q],
                }
            }
        except:
            pass

        # dissipation terms (one qubit, constant in time)
        t1 = model_params["T1"][q]
        g1 = 0 if t1 == 0 else 1.0 / t1  # * 2 * np.pi
        t2 = model_params["T2"][q]
        g2 = 0 if t2 == 0 else 1.0 / t2  # * 2 * np.pi

        # dissipation_dict["t1"].append((np.sqrt(g1 / 2), f"sp01_{q}", [q]))
        # dissipation_dict["t2"].append((np.sqrt(g2 / 2), f"Z01_{q}", [q]))
        # dissipation_dict["t1"].append((np.sqrt(g1 / 2), f"sp01_{ind}", [ind]))
        # dissipation_dict["t2"].append((np.sqrt(g2 / 2), f"Z01_{ind}", [ind]))
        dissipation_dict["t1"].append((np.sqrt(g1), f"sp01_{ind}", [ind]))
        dissipation_dict["t2"].append((np.sqrt(g2), f"Z01_{ind}", [ind]))

    # single coupler terms
    # for i, c in enumerate(couplers_list):
    for c in couplers_list:
        ind = platform_to_simulator_qubits[c]
        # drift Hamiltonian terms (constant in time)
        drift_hamiltonian_dict["one_body"].append(
            # (2 * np.pi * model_params["drive_freq"][c], f"O_{c}", [c])
            (2 * np.pi * model_params["drive_freq"][c], f"O_{ind}", [ind])
        )
        drift_hamiltonian_dict["one_body"].append(
            (
                np.pi * model_params["anharmonicity"][c],
                # f"O_{c} * O_{c} - O_{c}",
                # [c],
                f"O_{ind} * O_{ind} - O_{ind}",
                [ind],
            )
        )

        # drive Hamiltonian terms (amplitude determined by pulse sequence)
        # drive_hamiltonian_dict.update({f"D-{couplers_list[i]}": []})
        drive_hamiltonian_dict.update({f"D-{ind}": []})
        # drive_hamiltonian_dict[f"D-{couplers_list[i]}"].append(
        drive_hamiltonian_dict[f"D-{ind}"].append(
            # (2 * np.pi * model_params["rabi_freq"][c], f"X_{c}", [c])
            (2 * np.pi * model_params["rabi_freq"][c], f"X_{ind}", [ind])
        )

        # flux Hamiltonian terms (amplitude determined by processed pulse sequence)
        # flux_hamiltonian_dict.update({f"F-{couplers_list[i]}": []})
        flux_hamiltonian_dict.update({f"F-{ind}": []})
        # flux_hamiltonian_dict[f"F-{couplers_list[i]}"].append(
        flux_hamiltonian_dict[f"F-{ind}"].append(
            # (2 * np.pi, f"O_{c}", [c])
            (2 * np.pi, f"O_{ind}", [ind])
        )

        # flux detuning parameters:
        try:
            flux_params_dict |= {
                c: {
                    "flux_quanta": model_params["flux_quanta"][c],
                    "max_frequency": model_params["max_lo_freq"][c],
                    "current_frequency": model_params["drive_freq"][c],
                }
            }
        except:
            pass

        # dissipation terms for couplers
        t1 = model_params["T1"][c]
        g1 = 0 if t1 == 0 else 1.0 / t1  # * 2 * np.pi
        t2 = model_params["T2"][c]
        g2 = 0 if t2 == 0 else 1.0 / t2  # * 2 * np.pi

        # dissipation_dict["t1"].append((np.sqrt(g1 / 2), f"sp01_{c}", [c]))
        # dissipation_dict["t2"].append((np.sqrt(g2 / 2), f"Z01_{c}", [c]))
        # dissipation_dict["t1"].append((np.sqrt(g1 / 2), f"sp01_{ind}", [ind]))
        # dissipation_dict["t2"].append((np.sqrt(g2 / 2), f"Z01_{ind}", [ind]))
        dissipation_dict["t1"].append((np.sqrt(g1), f"sp01_{ind}", [ind]))
        dissipation_dict["t2"].append((np.sqrt(g2), f"Z01_{ind}", [ind]))

    # two-body terms (couplings)
    for key in list(
        model_params["coupling_strength"].keys()
    ):  # consistent with device_qubits_list
        ind2, ind1 = key.split(
            "_"
        )  # ind2 > ind1 with ind_qubit > ind_coupler as per Hilbert space ordering
        ind1 = platform_to_simulator_qubits[ind1]
        ind2 = platform_to_simulator_qubits[ind2]

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
        "device_qubits_list": device_qubits_list,
        "device_couplers_list": device_couplers_list,
        "qubits_list": qubits_list,
        "nlevels_q": nlevels_q,
        "couplers_list": couplers_list,
        "nlevels_c": nlevels_c,
        "drift": drift_hamiltonian_dict,
        "drive": drive_hamiltonian_dict,
        "flux": flux_hamiltonian_dict,
        "flux_params": flux_params_dict,
        "dissipation": dissipation_dict,
        "method": "master_equation",
        "readout_error": readout_error,
        "platform_to_simulator_qubits": platform_to_simulator_qubits,
        # "platform_to_simulator_channels": default_noflux_platform_to_simulator_channels(
        "platform_to_simulator_channels": default_platform_to_simulator_channels(
            device_qubits_list, device_couplers_list
        ),
    }

    return model_config
