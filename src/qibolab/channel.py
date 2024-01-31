from dataclasses import dataclass


@dataclass
class NamedChannel:
    """Channel that has a name. This is part of the end-user API, i.e. is in
    the top layer.

    The idea is the following:
        1. End users do not know what are the types of channels in a platform, they just know names.
           They use only NamedChannels to describe pulse sequences. They can create NamedChannels using the get_channel function below.
        2. User makes some assumptions about the types of channels. E.g. they assume that the channel NamedChannel("qubit_0/drive")
           is an IQ channel, hence they play IQ pulses on it, and they provide IQChannelConfig for it.
        3. Upon receival of the execution request qibolab validates that the requested execution can be done, i.e.
           channels with those names exist and user's assumptions are correct.
        4. qibolab proceeds with execution.
    For the last two steps qibolab needs to replace generic NamedChannels with concrete channels (e.g. ZurichIQChannel, QbloxDCChannel, etc.), and
    those should be available in the Platform description.

    TODO: I am not sure if having this class brings any benefit to this plan compared to the case where we just use naked str names, but I will figure
    this out later during implementation.

    TODO: One might argue that it is reasonable to provide the end user the types of channels as well, and then do all the validation while constructing the pulse
    sequence. I thought about this and failed to find real benefit, it just seems to complicate the code and the user-facing API for no real benefit.
    Please comment if you have anything to say regarding this.
    """

    name: str

    def __str__(self) -> str:
        return self.name


def get_channel(element: str, line: str) -> NamedChannel:
    """Named channel for given element (qubit|qubit|coupler|etc.), for given
    line (drive|flux|readout|etc.)

    This method can be used by users to get the channel that they are interested in, and then use this in their pulse sequence description.

    FIXME: the function signature is just a mock/sketch. Needs to be designed properly.
    """
    return NamedChannel(f"{element}/{line}")
