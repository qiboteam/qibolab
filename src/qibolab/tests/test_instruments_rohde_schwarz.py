import pathlib
import pytest
import yaml
from qibolab.instruments.rohde_schwarz import SGS100A


def load_runcard(name):
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / f"{name}.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    return settings


@pytest.mark.xfail
@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_sgs100a_init(device):
    settings = load_runcard("tiiq")    
    lo = SGS100A(**settings.get(f"LO_{device}_init_settings"))

    with pytest.raises(RuntimeError):
        frequency = lo.get_frequency()
    with pytest.raises(RuntimeError):
        power = lo.get_power()

    lo.close()


@pytest.mark.xfail
@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_sgs100a_setup(device):
    settings = load_runcard("tiiq")    
    lo = SGS100A(**settings.get(f"LO_{device}_init_settings"))

    power = settings.get(f"LO_{device}_settings").get("power")
    frequency = settings.get(f"LO_{device}_settings").get("frequency")
    lo.setup(power, frequency)

    assert lo.get_power() == power
    assert lo.get_frequency() == frequency

    lo.close()


@pytest.mark.xfail
@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_sgs100a_on_off(device):
    settings = load_runcard("tiiq")    
    lo = SGS100A(**settings.get(f"LO_{device}_init_settings"))

    power = settings.get(f"LO_{device}_settings").get("power")
    frequency = settings.get(f"LO_{device}_settings").get("frequency")
    lo.setup(power, frequency)

    lo.on()
    lo.off()

    lo.close()