from collections import OrderedDict
from typing import List, Optional

import numpy as np


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
