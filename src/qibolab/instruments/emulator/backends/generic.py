from collections import OrderedDict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Latex, display
from matplotlib.figure import Figure


def dec_to_basis_string(x: int, nlevels: list = [2]) -> list:
    """Converts an integer to a generalized bitstring in the computation basis
    of the full Hilbert space.

    Args:
        x (int): The integer (decimal number) to be converted.
        nlevels (list): The list of nlevels to convert the decimal number to. Defaults to [2].

    Returns:
        list: Generalized bitstring of x.
    """
    nqubits = len(nlevels)
    output_list = []
    y = x
    subdims_ = np.multiply.accumulate(nlevels)
    subdims = (subdims_[-1] / subdims_).astype(int)

    for sub_dim in subdims:
        coeff = np.divmod(y, sub_dim)[0]
        output_list.append(coeff)
        y -= coeff * sub_dim

    return output_list


def op_from_instruction(
    inst: tuple[float, str, list],
    op_dict: Optional[dict] = None,
    op_connectors_dict: Optional[dict] = None,
    multiply_coeff: bool = True,
):
    """Converts an instruction tuple into a quantum operator.

    Args:
        inst (tuple): The instruction tuple.
        op_dict (dict, optional): Dictionary mapping operator names to strings. Defaults to None.
        op_connectors_dict (dict, optional): Dictionary mapping operator connectors to operations. Defaults to None.
        multiply_coeff (bool): Multiplies coefficient to final quantum operator if True, or keep them separate in a tuple otherwise. Defaults to True.

    Returns:
        The quantum operator if multiply_coeff is True, or a tuple containing the coefficient and operator string otherwise.
    """
    coeff, s, op_qid_list = inst

    # construct op_dict and op_connectors_dict check dictionaries (involves strings and string operations)
    if op_dict is None:
        op_dict_check = {
            "b": {},
            "bdag": {},
            "O": {},
            "X": {},
            "Z01": {},
            "sp01": {},
        }
        for k in op_qid_list:
            op_dict_check["b"].update({k: f"op_b_{k}"})
            op_dict_check["bdag"].update({k: f"op_bdag_{k}"})
            op_dict_check["O"].update({k: f"op_O_{k}"})
            op_dict_check["X"].update({k: f"op_X_{k}"})
            op_dict_check["Z01"].update({k: f"op_Z01_{k}"})
            op_dict_check["sp01"].update({k: f"op_sp01_{k}"})
        op_dict = op_dict_check

    if op_connectors_dict is None:
        op_connectors_dict_check = OrderedDict(
            [
                ("^", lambda a, b: f"tensor({a},{b})"),
                ("*", lambda a, b: f"times({a},{b})"),
                ("+", lambda a, b: f"plus({a},{b})"),
                ("-", lambda a, b: f"minus({a},{b})"),
            ]
        )
        op_connectors_dict = op_connectors_dict_check

    def process_op(op_list: list, connector_list: List[str], connector: str) -> list:
        """Implements connectors between operators."""
        index_list = []
        for i in range(connector_list.count(connector)):
            ind = connector_list.index(connector)
            index_list.append(ind + len(index_list))
            connector_list.pop(ind)

        new_op_list = []
        for i, op in enumerate(op_list):
            if i in index_list:
                next_op = op_list[i + 1]
                if i - 1 in index_list:
                    new_op_list[-1] = op_connectors_dict[connector](
                        new_op_list[-1], next_op
                    )
                else:
                    new_op_list.append(op_connectors_dict[connector](op, next_op))
            elif i - 1 in index_list:
                pass
            else:
                new_op_list.append(op)
        return new_op_list, connector_list

    # convert operator instruction strings into list of components
    split_inst = s.split(" ")
    op_list = []
    # implement operators
    for op_key in split_inst[::2]:
        op_name, qid = op_key.split("_")
        op_list.append(op_dict[op_name][qid])
    # implement operator connections
    connector_list = split_inst[1::2]
    for connector in list(op_connectors_dict.keys()):
        op_list, connector_list = process_op(op_list, connector_list, connector)

    if multiply_coeff:
        return coeff * op_list[0]
    else:
        return coeff, op_list[0]


def print_hamiltonian(model_config, op_qid_list: list = None):
    """Prints Hamiltonian the model configuration.

    Args:
        model_config (dict): Model configuration dictionary.
        op_qid_list (list, optional): List of qubit/coupler IDs present in the model. If None, extracts this information from model_config. Defaults to None.
    """

    def format_instruction(op_instruction, is_dissipation=False):
        """Reformats op_instructions."""
        if is_dissipation:
            ghz_units = "\\sqrt{{ \\text{GHz} }}"
            return [
                f"${op_instruction[1]}$",
                f"${op_instruction[0]/np.sqrt(2*np.pi)}~{ghz_units}$",
            ]
        else:
            ghz_units = "\\text{GHz}"
            return [
                f"${op_instruction[1]}$",
                f"${op_instruction[0]/2/np.pi}~{ghz_units}$",
            ]

    if op_qid_list is None:
        qubits_list = model_config["qubits_list"]
        couplers_list = model_config["couplers_list"]
        op_qid_list = qubits_list + couplers_list

    latex_op_dict = {
        "b": {},
        "bdag": {},
        "O": {},
        "X": {},
        "Z01": {},
        "sp01": {},
    }
    for k in op_qid_list:
        latex_op_dict["b"].update({k: f"b_{k}"})
        latex_op_dict["bdag"].update({k: rf"b^{{\dagger}}_{k}"})
        latex_op_dict["O"].update({k: f"O_{k}"})
        latex_op_dict["X"].update({k: f"X_{k}"})
        latex_op_dict["Z01"].update({k: rf"\sigma^Z_{k}"})
        latex_op_dict["sp01"].update({k: rf"\sigma^+_{k}"})

    latex_op_connectors_dict = OrderedDict(
        [
            ("^", lambda a, b: rf"{a}{{\otimes}}{b}"),
            ("*", lambda a, b: f"{a}{b}"),
            ("+", lambda a, b: f"{a}+{b}"),
            ("-", lambda a, b: f"{a}-{b}"),
        ]
    )

    basic_dict = [
        rf"$O_i = b^{{\dagger}}_i b_i$",
        rf"$X_i = b^{{\dagger}}_i + b_i$",
    ]
    print("Dictionary")
    for i in basic_dict:
        display(Latex(i))
    print("\n")
    print("-" * 21)

    print("One-body drift terms:")
    print("-" * 21)
    for op_instruction in model_config["drift"]["one_body"]:
        op_instruction = op_from_instruction(
            op_instruction,
            latex_op_dict,
            latex_op_connectors_dict,
            multiply_coeff=False,
        )
        inst = format_instruction(op_instruction)
        display(Latex(inst[0]))
        display(Latex(inst[1]))
    print("-" * 21)

    print("Two-body drift terms:")
    print("-" * 21)
    if len(model_config["drift"]["two_body"]) == 0:
        print("None")
    for op_instruction in model_config["drift"]["two_body"]:
        op_instruction = op_from_instruction(
            op_instruction,
            latex_op_dict,
            latex_op_connectors_dict,
            multiply_coeff=False,
        )
        inst = format_instruction(op_instruction)
        display(Latex(inst[0]))
        display(Latex(inst[1]))
    print("-" * 21)

    print("One-body drive terms:")
    print("-" * 21)
    for drive_instructions in model_config["drive"].values():
        for op_instruction in drive_instructions:
            op_instruction = op_from_instruction(
                op_instruction,
                latex_op_dict,
                latex_op_connectors_dict,
                multiply_coeff=False,
            )
            inst = format_instruction(op_instruction)
            display(Latex(inst[0]))
            display(Latex(inst[1]))
    print("-" * 21)

    print("Dissipative terms:")
    print("-" * 21)
    for key in model_config["dissipation"].keys():
        print(">>", key, "Linblad operators:")
        if len(model_config["dissipation"][key]) == 0:
            print("None")
        for op_instruction in model_config["dissipation"][key]:
            op_instruction = op_from_instruction(
                op_instruction,
                latex_op_dict,
                latex_op_connectors_dict,
                multiply_coeff=False,
            )
            inst = format_instruction(op_instruction, is_dissipation=True)
            display(Latex(inst[0]))
            display(Latex(inst[1]))
    print("-" * 21)


def plot_fidelity_history(
    time_list: list,
    fid_list_all: list,
    labels: list,
    time_in_dt: bool = False,
) -> Figure:
    """Plots the fidelity history.

    Args:
        time_list (list): Specifies the list of times used in simulation.
        fid_list_all (list): List of fidelity histories.
        labels (list): List of labels. Displayed in the plot legend.
        time_in_dt (bool): Specify the units of the x-axis in the plots to be in dt (inverse sampling rate) if True and in ns if False. Defaults to False.

    Returns:
        Figure: Figure of input fidelity histories with labels.
    """

    fig = plt.figure()
    for result_ind in range(len(labels)):
        plt.plot(
            time_list,
            fid_list_all[result_ind],
            label=labels[result_ind],
        )
    plt.legend(loc="upper left")
    plt.ylabel("Overlap with basis state")
    if time_in_dt:
        plt.xlabel("Time / dt")
    else:
        plt.xlabel("Time / ns")
    for result_ind in range(len(labels)):
        print(
            labels[result_ind],
            fid_list_all[result_ind][0],
            fid_list_all[result_ind][-1],
        )

    return fig
