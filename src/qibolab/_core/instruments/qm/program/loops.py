import warnings
from typing import Union

import numpy as np
import numpy.typing as npt
from qm.qua import Cast, Variable


def from_array(var: Variable, array: Union[npt.NDArray, list]):
    """Function parametrizing the QUA `for_` loop from a python array.

    Taken from qualang_tools to avoid the dependency.

    Args:
        var: the QUA variable that will be looped over (int or fixed).
        array: a Python list or numpy array containing the values over which `a` will be looping.
            The spacing must be even in linear or logarithmic scales and it cannot be a QUA array.

    Returns:
        QUA for_ loop parameters (var, init, cond, update) as defined in https://qm-docs.qualang.io/api_references/qua/dsl_main?highlight=for_#qm.qua._dsl.for_.
    """

    # Check for array length
    if len(array) == 0:
        raise Exception("The array must be of length > 0.")
    elif len(array) == 1:
        return var, array[0], var <= array[0], var + 1
    # Check QUA vs python variables
    if not isinstance(var, Variable):
        raise Exception("The first argument must be a QUA variable.")
    if (not isinstance(array[0], (np.generic, int, float))) or (
        isinstance(array[0], bool)
    ):
        raise Exception("The array must be an array of python variables.")
    # Check array increment
    if np.isclose(np.std(np.diff(array)), 0):
        increment = "lin"
    elif np.isclose(np.std(array[1:] / array[:-1]), 0, atol=1e-3):
        increment = "log"
    else:
        raise Exception(
            "The spacing of the input array must be even in linear or logarithmic scales. Please use `for_each_()` for arbitrary scans."
        )

    # Get the type of the specified QUA variable
    start = array[0]
    stop = array[-1]
    # Linear increment
    if increment == "lin":
        step = array[1] - array[0]

        if var.is_int():
            # Check that the array is an array of int with integer increments
            if (
                not float(step).is_integer()
                or not float(start).is_integer()
                or not float(stop).is_integer()
            ):
                raise Exception(
                    "When looping over a QUA int variable, the step and array elements must be integers."
                )
            # Generate the loop parameters for positive and negative steps
            if step > 0:
                return var, int(start), var <= int(stop), var + int(step)
            else:
                return var, int(start), var >= int(stop), var + int(step)

        elif var.is_fixed():
            # Check for fixed number overflows
            if not (-8 <= start < 8) and not (-8 <= stop < 8):
                raise Exception("fixed numbers are bounded to [-8, 8).")

            # Generate the loop parameters for positive and negative steps
            if step > 0:
                return (
                    var,
                    float(start),
                    var < float(stop) + float(step) / 2,
                    var + float(step),
                )
            else:
                return (
                    var,
                    float(start),
                    var > float(stop) + float(step) / 2,
                    var + float(step),
                )
        else:
            raise Exception(
                "This variable type is not supported. Please use a QUA 'int' or 'fixed' or contact a QM member for assistance."
            )
    # Logarithmic increment
    elif increment == "log":
        step = array[1] / array[0]

        if var.is_int():
            warnings.warn(
                "When using logarithmic increments with QUA integers, the resulting values will slightly differ from the ones in numpy.logspace() because of rounding errors. \n Please use the get_equivalent_log_array() function to get the exact values taken by the QUA variable and note that the number of points may also change."
            )
            if step > 1:
                if int(round(start) * float(step)) == int(round(start)):
                    raise ValueError(
                        "Two successive values in the scan are equal after being cast to integers which will make the QUA for_ loop fail. \nEither increase the logarithmic step or use for_each_(): https://docs.quantum-machines.co/1.1.6/qm-qua-sdk/docs/Guides/features/?h=for_ea#for_each."
                    )
                else:
                    return (
                        var,
                        round(start),
                        var < round(stop) * np.sqrt(float(step)),
                        Cast.mul_int_by_fixed(var, float(step)),
                    )
            else:
                return (
                    var,
                    round(start),
                    var > round(stop) / np.sqrt(float(step)),
                    Cast.mul_int_by_fixed(var, float(step)),
                )

        elif var.is_fixed():
            # Check for fixed number overflows
            if not (-8 <= start < 8) and not (-8 <= stop < 8):
                raise Exception("fixed numbers are bounded to [-8, 8).")

            if step > 1:
                return (
                    var,
                    float(start),
                    var < float(stop) * np.sqrt(float(step)),
                    var * float(step),
                )
            else:
                return (
                    var,
                    float(start),
                    var > float(stop) * np.sqrt(float(step)),
                    var * float(step),
                )
        else:
            raise Exception(
                "This variable type is not supported. Please use a QUA 'int' or 'fixed' or contact a QM member for assistance."
            )
