import numpy as np


def print_readable_snapshot(
    device, file, update: bool = False, max_chars: int = 80
) -> None:
    """Prints a readable version of the snapshot. The readable snapshot
    includes the name, value and unit of each parameter. A convenience function
    to quickly get an overview of the status of an instrument.

    Args:
        update: If ``True``, update the state by querying the
            instrument. If ``False``, just use the latest values in memory.
            This argument gets passed to the snapshot function.
        max_chars: the maximum number of characters per line. The
            readable snapshot will be cropped if this value is exceeded.
            Defaults to 80 to be consistent with default terminal width.
    """
    floating_types = (float, np.integer, np.floating)
    snapshot = device.snapshot(update=update)

    par_lengths = [len(p) for p in snapshot["parameters"]]

    # Min of 50 is to prevent a super long parameter name to break this
    # function
    par_field_len = min(max(par_lengths) + 1, 50)

    file.write(device.name + ":" + "\n")
    file.write("{0:<{1}}".format("\tparameter ", par_field_len) + "value" + "\n")
    file.write("-" * max_chars + "\n")
    for par in sorted(snapshot["parameters"]):
        name = snapshot["parameters"][par]["name"]
        msg = "{0:<{1}}:".format(name, par_field_len)

        # in case of e.g. ArrayParameters, that usually have
        # snapshot_value == False, the parameter may not have
        # a value in the snapshot
        val = snapshot["parameters"][par].get("value", "Not available")

        unit = snapshot["parameters"][par].get("unit", None)
        if unit is None:
            # this may be a multi parameter
            unit = snapshot["parameters"][par].get("units", None)
        if isinstance(val, floating_types):
            msg += f"\t{val:.5g} "
            # numpy float and int types format like builtins
        else:
            msg += f"\t{val} "
        if unit != "":  # corresponds to no unit
            msg += f"({unit})"
        # Truncate the message if it is longer than max length
        if len(msg) > max_chars and not max_chars == -1:
            msg = msg[0 : max_chars - 3] + "..."
        file.write(msg + "\n")

    for submodule in device.submodules.values():
        print_readable_snapshot(submodule, file, update=update, max_chars=max_chars)


#
