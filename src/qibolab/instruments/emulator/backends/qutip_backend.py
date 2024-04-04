"""
qutip_backend.py
----------------
This module provides a backend for Quantum Toolbox in Python (QuTiP) to simulate QPUs.
"""

from collections import OrderedDict
from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
from qutip import Options, Qobj, basis, expect, ket2dm, mesolve, ptrace
from qutip.operators import identity as Id
from qutip.tensor import tensor
from qutip.ui.progressbar import EnhancedTextProgressBar

from qibolab.instruments.emulator.backends.generic import (
    dec_to_basis_string,
    op_from_instruction,
)


def get_default_qutip_sim_opts():
    """Returns the default simulation options for the Qutip backend.

    Returns:
        Options: The default simulation options.
    """
    sim_opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
    sim_opts.normalize_output = False  # mesolve is x3 faster if this is False

    return sim_opts


class Qutip_Simulator:
    """Builds pulse simulator components in qutip backend. Pulse simulation is
    implemented by `qevolve` method. Qutip_Simulator does not interact with
    Qibolab.

    Note that objects have either little or big endian order.
    Little endian: decreasing order of qubit/coupler index (smallest index = 0).
    Big endian: increasing order of qubit index/coupler index.
    Full system Hilbert space structure: little endian, qubits first then couplers.
    """

    def __init__(self, model_config: dict, sim_opts: Optional[Options] = None):
        """Initializes with qutip simulation backend with model_config and
        qutip simulation options.

        Args:
            model_config (dict): Model configuration dictionary.
            sim_opts (`qutip.Options`, optional): Qutip simulation options. If None, default
            options are used.
        """
        self.model_config = model_config
        if sim_opts is None:
            sim_opts = get_default_qutip_sim_opts()
        self.sim_opts = sim_opts
        self.update()

    def update_sim_opts(self, updated_sim_opts: Options):
        self.sim_opts = updated_sim_opts

    def update(self):
        """Updates the simulation backend by loading all parameters from
        `self.model_config` and `self.sim_opts`."""
        ### from model_config ###
        self.nlevels_q = self.model_config["nlevels_q"]  # as per runcard, big endian
        self.nlevels_c = self.model_config["nlevels_c"]  # as per runcard, big endian
        self.nlevels_HS = np.flip(
            self.nlevels_c + self.nlevels_q
        ).tolist()  # little endian, qubits first then couplers
        self.qubits_list = self.model_config[
            "qubits_list"
        ]  # as per runcard, big endian
        self.couplers_list = self.model_config[
            "couplers_list"
        ]  # as per runcard, big endian
        self.combined_list = (
            self.couplers_list + self.qubits_list
        )  # as per runcard, big endian
        self.HS_list = np.flip(self.combined_list)
        print("Hilbert space structure: ", self.HS_list.tolist())
        print("Hilbert space dimensions: ", self.nlevels_HS)

        self.topology = self.model_config["topology"]
        self.nqubits = len(self.qubits_list)
        self.ncouplers = len(self.couplers_list)
        self.sim_method = self.model_config["method"]

        ### derived parameters ###
        self.qid_nlevels_map = {}  # coupler and qubit cannot have the same label
        for i, c in enumerate(self.couplers_list):
            self.qid_nlevels_map.update({c: self.nlevels_c[i]})
        for i, q in enumerate(self.qubits_list):
            self.qid_nlevels_map.update({q: self.nlevels_q[i]})

        ### n-level qubit/coupler base operators ###
        self.op_dict = {"basis": {}, "sig": {}, "pm1_matrix": {}, "id": {}}
        for qid, nlevels in self.qid_nlevels_map.items():
            basis_list = [basis(nlevels, i) for i in range(nlevels)]
            sig = [
                [basis_list[i] * basis_list[j].dag() for j in range(nlevels)]
                for i in range(nlevels)
            ]
            self.op_dict["basis"].update({qid: basis_list})
            self.op_dict["sig"].update({qid: sig})
            self.op_dict["pm1_matrix"].update(
                {qid: [(sig[i][i + 1] + sig[i + 1][i]) / 2 for i in range(nlevels - 1)]}
            )
            Id = sig[0][0]
            for i in range(1, nlevels):
                Id += sig[i][i]
            self.op_dict["id"].update({qid: Id})

        ### operator connector dictionary ###
        self.op_connectors_dict = OrderedDict(
            [
                ("^", lambda a, b: tensor(a, b)),
                ("*", lambda a, b: a * b),
                ("+", lambda a, b: a + b),
                ("-", lambda a, b: a - b),
            ]
        )

        ### multi-qubit qubit-coupler up-state ###
        self.basis_list = self.op_dict["basis"]
        self.psi0 = Qobj(1)
        for i, qind in enumerate(self.HS_list):
            if i == 0:
                self.psi0 = self.basis_list[qind][0]
            else:
                self.psi0 = tensor(self.psi0, self.basis_list[qind][0])

        self.H = []
        self.pulse_sim_time = []
        self.pulse_sim_history = []
        self.pulse_sim_time_list = []

        ### build dictionary of base qutip operators ###
        self.op_dict.update(
            {
                "b": {},
                "bdag": {},
                "O": {},
                "X": {},
                "Z01": {},
                "sp01": {},
            }
        )

        ### Construct qutip op_dict for each qubit and coupler ###
        for qid, nlevels in self.qid_nlevels_map.items():
            sig = self.op_dict["sig"][qid]
            b = sig[0][1]
            for i in range(nlevels - 2):
                b += np.sqrt(i + 2) * sig[i + 1][i + 2]
            bdag = b.dag()
            O = bdag * b
            X = b + bdag
            Z01 = sig[0][0] - sig[1][1]
            sp01 = sig[0][1]

            self.op_dict["b"].update({qid: b})
            self.op_dict["bdag"].update({qid: bdag})
            self.op_dict["O"].update({qid: O})
            self.op_dict["X"].update({qid: X})
            self.op_dict["Z01"].update({qid: Z01})
            self.op_dict["sp01"].update({qid: sp01})

        ## initialize operators ##
        self.drift = Qobj(dims=[self.nlevels_HS, self.nlevels_HS])
        self.operators = {}
        self.static_dissipators = []

        ### drift ###
        for op_instruction in self.model_config["drift"]["one_body"]:
            self.drift += self.make_operator(op_instruction)
        for op_instruction in self.model_config["drift"]["two_body"]:
            self.drift += self.make_operator(op_instruction)

        ### drive ###
        for channel_name, op_instruction_list in self.model_config["drive"].items():
            channel_op = Qobj(dims=[self.nlevels_HS, self.nlevels_HS])
            for op_instruction in op_instruction_list:
                channel_op += self.make_operator(op_instruction)
            self.operators.update({channel_name: channel_op})

        ### dissipation ###
        for op_instruction in self.model_config["dissipation"]["t1"]:
            self.static_dissipators += [self.make_operator(op_instruction)]
        for op_instruction in self.model_config["dissipation"]["t2"]:
            self.static_dissipators += [self.make_operator(op_instruction)]

    def make_arbitrary_state(
        self, statedata: np.ndarray, is_qibo_state_vector: bool = False
    ) -> Qobj:
        """Creates a quantum state object of the full system Hilbert space
        using the given state data.

        Args:
            statedata (np.ndarray): The state data, in little endian order for compatibility with pulse simulation.
            is_qibo_state_vector (bool): Flag to change statedata from big endian (qibo convention) to little endian order.

        Returns:
            `qutip.Qobj`: The quantum state object.
        """
        if len(statedata.shape) == 1:  # statevector
            dims = [self.nlevels_HS, np.ones(len(self.nlevels_HS), dtype=int).tolist()]
        else:  # density matrix
            dims = [self.nlevels_HS, self.nlevels_HS]
        arbitrary_state = make_arbitrary_state(statedata, dims)
        if is_qibo_state_vector is True:
            arbitrary_state = self.flip_HS(arbitrary_state)

        return arbitrary_state

    def extend_op_dim(
        self, op_qobj: Qobj, op_indices_q: List[int] = [0], op_indices_c: List[int] = []
    ) -> Qobj:
        """Extends the dimension of an operator from its local Hilbert space to
        the full system Hilbert space.

        Args:
            op_qobj (`qutip.Qobj`): The quantum object representation of the operator in the local Hilbert space.
            op_indices_q (list): List of qubit indices involved in the operator, in little endian order.
            op_indices_c (list): List of coupler indices involved in the operator, in little endian order.

        Returns:
            `qutip.Qobj`: The quantum object representation of the operator in the full system Hilbert space.
        """
        return extend_op_dim(
            op_qobj,
            op_indices_q,
            op_indices_c,
            nlevels_q=self.nlevels_q,
            nlevels_c=self.nlevels_c,
        )

    def make_operator(self, op_instruction) -> Qobj:
        """Constructs the operator specified by op_instruction as a Qobj and
        extends it to the full system Hilbert space.

        Args:
            op_instruction (tuple): The instruction tuple containing the coefficient, operator string, and a list of qubit IDs that the operator acts on. The operator string and the qubit ID list are required to be in little endian order, and should have consistent qubit IDs.

        Returns:
            `qutip.Qobj`: The quantum object representation of the operator in the full system Hibert space.
        """
        coeff, s, op_qid_list = op_instruction
        op_localHS = op_from_instruction(
            op_instruction,
            op_dict=self.op_dict,
            op_connectors_dict=self.op_connectors_dict,
        )

        op_indices_q = []
        op_indices_c = []
        for k in op_qid_list:
            if k in self.qubits_list:
                op_indices_q.append(self.qubits_list.index(k))
            if k in self.couplers_list:
                op_indices_c.append(self.couplers_list.index(k))

        op_fullHS = self.extend_op_dim(
            op_localHS, op_indices_q=op_indices_q, op_indices_c=op_indices_c
        )
        return op_fullHS

    def flip_HS(self, state: Qobj):
        """Changes state from little endian ordering (qubits, couplers) to big
        endian (qubits, couplers) and vice versa, while retaining the same
        Hilbert space structure with qubits first followed by couplers."""
        nqubits_total = self.nqubits + self.ncouplers
        flipped_q_indices = np.flip(range(nqubits_total)[: self.nqubits])
        flipped_c_indices = np.flip(range(nqubits_total)[self.nqubits :])
        reordering_flip = np.append(flipped_q_indices, flipped_c_indices).astype(int)

        return state.permute(reordering_flip)

    def qevolve(
        self,
        channel_waveforms: dict,
        simulate_dissipation: bool = False,
    ) -> tuple[np.ndarray, List[int]]:
        """Performs the quantum dynamics simulation.

        Args:
            channel_waveforms (dict): The dictionary containing the list of discretized time steps and the corresponding channel waveform amplitudes labelled by the respective channel names.
            simulate_dissipation (bool): Flag to add (True) or not (False) the dissipation terms associated with T1 and T2 times.

        Returns:
            tuple: A tuple containing the reduced density matrix of the quantum state at the end of simulation in the Hilbert space specified by the qubits present in the readout channels (little endian), as well as the corresponding list of qubit indices.
        """
        full_time_list = channel_waveforms["time"]
        channel_names = list(channel_waveforms["channels"].keys())

        fp_list = []
        for channel_name in channel_names:
            fp_list.append(
                function_from_array(
                    channel_waveforms["channels"][channel_name], full_time_list
                )
            )

        drift = self.drift
        scheduled_operators = []
        ro_qubit_list = []

        # add corresponding operators for non-readout channels to scheduled_operators; add qubit indices of readout channels to ro_qubit_list
        for channel_name in channel_names:
            if channel_name[:2] != "R-":
                scheduled_operators.append(self.operators[channel_name])
            else:
                ro_qubit_list.append(int(channel_name[2:]))
        ro_qubit_list = np.flip(np.sort(ro_qubit_list))

        if simulate_dissipation is True:
            static_dissipators = self.static_dissipators
        else:
            static_dissipators = []

        H = [drift]
        for i, op in enumerate(scheduled_operators):
            H.append([op, fp_list[i]])
        self.H.append(H)

        sim_start_time = timer()
        if self.sim_method == "master_equation":
            result = mesolve(
                H,
                self.psi0,
                full_time_list,
                c_ops=static_dissipators,
                options=self.sim_opts,
                progress_bar=EnhancedTextProgressBar(
                    len(full_time_list), int(len(full_time_list) / 100)
                ),
            )

        sim_end_time = timer()
        sim_time = sim_end_time - sim_start_time

        final_state = result.states[-1]
        # saves history of states (in little endian, opposite to qibo convention)
        self.pulse_sim_history.append(result.states)
        self.pulse_sim_time_list.append(full_time_list)
        self.pulse_sim_time.append(sim_time)
        print("simulation time", sim_time)

        return self.qobj_to_reduced_dm(final_state, ro_qubit_list)

    def qobj_to_reduced_dm(
        self, emu_qstate: Qobj, qubit_list: List[int]
    ) -> tuple[np.ndarray, List[int]]:
        """Computes the reduced density matrix of the emulator quantum state
        specified by `qubit_list`.

        Args:
            emu_qstate (`qutip.Qobj`): Quantum state (full system Hilbert space).
            qubit_list (list): List of target qubit indices to keep in the reduced density matrix. Order of qubit indices is not important.

        Returns:
            tuple: The resulting reduced density matrix and the little endian ordered list of qubit indices specifying the Hilbert space of the reduced density matrix.
        """
        hilbert_space_ind_list = []
        for qubit_ind in qubit_list:
            hilbert_space_ind_list.append(
                self.nqubits - 1 - qubit_ind
            )  # based on little endian order of HS, qubits only; independent of couplers
        reduced_dm = ptrace(emu_qstate, hilbert_space_ind_list).full()
        rdm_qubit_list = np.flip(np.sort(qubit_list)).tolist()

        return reduced_dm, rdm_qubit_list

    def state_from_basis_vector(
        self, basis_vector: List[int], cbasis_vector: List[int] = None
    ) -> Qobj:
        """Constructs the corresponding computational basis state of the
        generalized Hilbert space specified by qubit_list.

        Args:
            basis_vector (List[int]): Generalized bitstring that specifies the computational basis state corresponding to the qubits in big endian order.
            cbasis_vector (List[int]): Generalized bitstring that specifies the computational basis state corresponding to the couplers in big endian order.

        Returns:
            `qutip.Qobj`: Computational basis state consistent with Hilbert space
            structure: little endian, qubits first then couplers.
        Raises:
            Exception: If the length of basis_vector is not equal to `self.nqubits` or the length of cbasis_vector is not equal to `self.ncouplers`.
        """
        # checks
        if len(basis_vector) != self.nqubits:
            raise Exception("length of basis_vector does not match number of qubits!")
        if cbasis_vector is None:
            cbasis_vector = [0 for c in range(self.ncouplers)]
        else:
            if len(cbasis_vector) != self.ncouplers:
                raise Exception(
                    "length of cbasis_vector does not match number of couplers!"
                )

        basis_list = self.op_dict["basis"]
        fullstate = Qobj(1)

        combined_basis_vector = (
            cbasis_vector + basis_vector
        )  # basis_vector + cbasis_vector #
        for ind, coeff in enumerate(combined_basis_vector):
            qind = self.combined_list[ind]
            fullstate = tensor(
                basis_list[qind][coeff], fullstate
            )  # constructs little endian HS, qubits first then couplers, as per evolution

        return fullstate

    def fidelity_history(
        self,
        sim_index: int = -1,
        reference_states: Optional[list] = None,
        labels: list = None,
        show_plot: bool = True,
        time_in_dt: bool = False,
    ) -> List[Qobj]:
        """Calculates the fidelity history, i.e. the overlaps between the
        device quantum state at each time step in the pulse simulation with
        respect to the input reference states.

        Args:
            sim_index (int, optional): Specifies the pulse sequence simulations stored in the Simulation backend of interest. Defaults to -1, i.e. the last simulation.
            reference_states (list, optional): List of reference states to compare the pulse simulation with.
            If not provided, all computational basis states will be assigned.
            labels (list, optional): List of labels for the reference states. Displayed in the plot legend.
            If not provided, the method will generate labels based on the basis states.
            show_plot (bool): Whether to show the fidelity history plot. Defaults to True.
            time_in_dt (bool): Specify the units of the x-axis in the plots to be in dt (inverse sampling rate) if True and in ns if False. Defaults to False.

        Returns:
            list: List of fidelity histories for each reference state.
        """
        fid_list_all = []
        if reference_states is None:
            reference_states = []
            labels = []

            full_HS_dim = np.product(self.nlevels_HS)
            for state_id in range(full_HS_dim):
                basis_string = dec_to_basis_string(state_id, self.nlevels_HS)
                labels.append(basis_string)
                basis_state = self.state_from_basis_vector(
                    basis_string[: self.nqubits], basis_string[self.nqubits :]
                )
                psi0 = ket2dm(basis_state)
                reference_states.append(psi0)

        total_samples = len(self.pulse_sim_time_list[sim_index])
        for ref_state in reference_states:
            fid_list = [
                expect(ref_state, self.pulse_sim_history[sim_index][i])
                for i in range(total_samples)
            ]
            fid_list_all.append(fid_list)

        if show_plot is True:
            import matplotlib.pyplot as plt

            plt.figure()
            for result_ind in range(len(labels)):
                plt.plot(
                    self.pulse_sim_time_list[sim_index],
                    fid_list_all[result_ind],
                    label=labels[result_ind],
                )
            plt.legend(loc="upper left")
            plt.ylabel("Overlap with basis state")
            if time_in_dt:
                plt.xlabel("Time / dt")
            else:
                plt.xlabel("Time / ns")
            # plt.show()
            for result_ind in range(len(labels)):
                print(
                    labels[result_ind],
                    fid_list_all[result_ind][0],
                    fid_list_all[result_ind][-1],
                )

        return fid_list_all


def make_arbitrary_state(statedata: np.ndarray, dims: list[int]) -> Qobj:
    """Create a quantum state object using the given state data and dimensions.

    Args:
        statedata (np.ndarray): The state data.
        dims (list): The dimensions of the state.

    Returns:
        `qutip.Qobj`: The quantum state object.
    """
    shape = (np.product(dims[0]), np.product(dims[1]))
    if shape[1] == 1:
        statetype = "ket"
    elif shape[0] == shape[1]:
        statetype = "oper"

    return Qobj(statedata, dims=dims, shape=shape, type=statetype)


def function_from_array(y: np.ndarray, x: np.ndarray):
    """Return function given a data array y and time array x."""

    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same first dimension")

    yx = np.column_stack((y, x))
    yx = yx[yx[:, -1].argsort()]

    def func(t, args):
        idx = np.searchsorted(yx[1:, -1], t, side="right")
        return yx[idx, 0]

    return func


def extend_op_dim(
    op_qobj: Qobj,
    op_indices_q: List[int] = [0],
    op_indices_c: List[int] = [],
    nlevels_q: List[int] = [2],
    nlevels_c: List[int] = [],
) -> Qobj:
    """Extenda the dimension of the input operator from its local Hilbert space
    to a larger n-body Hilbert space.

    Args:
        op_qobj (`qutip.Qobj`): The quantum object representation of the operator in the local Hilbert space.
        op_indices_q (list): List of qubit indices involved in the operator, in little endian order. Defaults to [0].
        op_indices_c (list): List of coupler indices involved in the operator, in little endian order. Defaults to [].
        nlevels_q (list): List of the number of levels for each qubit, in big endian order. Defaults to [2].
        nlevels_c (list): List of the number of levels for each coupler, in big endian order. Defaults to [].

    Returns:
        `qutip.Qobj`: The quantum object representation of the operator in the Hilbert space with extended dimensions. Hilbert space structure: little endian, qubits first then couplers.

    Raises:
        Exception: If the length of op_qobj.dims[0] does not match the sum of the lengths of op_indices_q and op_indices_c, or if nlevels_q and nlevels_c inputs do not support op_indices_q and op_indices_c, or if dimensions of local Hilbert space operator do not match the nlevels of op_indices_q and op_indices_c.
    """

    ncouplers = len(nlevels_c)
    nqubits = len(nlevels_q)
    nlevels_cq = nlevels_c + nlevels_q  # reverse order from Hilbert space structure
    op_indices_q_shifted = [ind + ncouplers for ind in op_indices_q]
    op_indices_full = op_indices_q_shifted + op_indices_c

    full_index_list = np.flip(range(nqubits + ncouplers))
    missing_indices = list(full_index_list.copy())
    for ind in full_index_list:
        if ind in op_indices_full:
            pos = np.where(ind == missing_indices)[0][0]
            missing_indices.pop(pos)

    unordered_index_list = op_indices_full + missing_indices

    # checks
    if len(op_qobj.dims[0]) != len(op_indices_q) + len(op_indices_c):
        raise Exception("op indices and op mismatch!")
    try:
        nlevels_HS_local = []
        for ind_q in op_indices_q:
            nlevels_HS_local.append(nlevels_q[ind_q])
        for ind_c in op_indices_c:
            nlevels_HS_local.append(nlevels_c[ind_c])
    except:
        raise Exception("op indices dim and nlevels mismatch!")
    for ind, nlevel in enumerate(nlevels_HS_local):
        if nlevel != op_qobj.dims[0][ind]:
            raise Exception(f"mismatch in op dim: index {ind}!")

    full_qobj = op_qobj
    for ind in missing_indices:
        full_qobj = tensor(full_qobj, Id(nlevels_cq[ind]))

    inverse_qubit_order = np.flip(np.argsort(unordered_index_list))

    return full_qobj.permute(inverse_qubit_order)
