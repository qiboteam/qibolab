from pydantic import model_validator

from qibolab.serialize_ import Model

Register = int
Immediate = int
Value = Register | Immediate


class Illegal(Model):
    """"""


class Stop(Model):
    """"""


class Nop(Model):
    """"""


Control = Illegal | Stop | Nop


class Jmp(Model):
    address: Value


class Jge(Model):
    a: Register
    b: Immediate
    address: Value


class Jlt(Model):
    a: Register
    b: Immediate
    address: Value


class Loop(Model):
    a: Register
    address: Value


Jump = Jmp | Jge | Jlt | Loop


class Move(Model):
    source: Value
    destination: Register


class Not(Model):
    source: Value
    destination: Register


class Add(Model):
    a: Register
    b: Value
    destination: Register


class Sub(Model):
    a: Register
    b: Value
    destination: Register


class And(Model):
    a: Register
    b: Value
    destination: Register


class Or(Model):
    a: Register
    b: Value
    destination: Register


class Xor(Model):
    a: Register
    b: Value
    destination: Register


class Asl(Model):
    a: Register
    b: Value
    destination: Register


class Asr(Model):
    a: Register
    b: Value
    destination: Register


Arithmetic = Move | Not | Add | Sub | And | Or | Xor | Asl | Asr


class SetMrk(Model):
    mask: Value


class SetFreq(Model):
    value: Value


class ResetPh(Model):
    """"""


class SetPh(Model):
    value: Value


class SetPhDelta(Model):
    value: Value


class SetAwgGain(Model):
    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) == type(self.value_1)
        return self


class SetAwgOffs(Model):
    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) == type(self.value_1)
        return self


ParamOps = SetMrk | SetFreq | ResetPh | SetPh | SetPhDelta | SetAwgGain | SetAwgOffs

Q1Instr = Control | Jump | Arithmetic | ParamOps


class SetCond(Model):
    enable: Value
    mask: Value
    operator: Value
    else_duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.enable) == type(self.mask)
        assert type(self.enable) == type(self.operator)
        return self


Conditional = SetCond


class UpdParam(Model):
    duration: Immediate


class Play(Model):
    wave_0: Value
    wave_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.wave_0) == type(self.wave_1)
        return self


class Acquire(Model):
    acquisition: Immediate
    bin: Value
    duration: Immediate


class AcquireWeighed(Model):
    acquisition: Immediate
    bin: Value
    weight_0: Value
    weight_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.bin) == type(self.weight_0)
        assert type(self.bin) == type(self.weight_1)
        return self


class AcquireTtl(Model):
    acquisition: Immediate
    bin: Value
    enable: Immediate
    duration: Immediate


Io = UpdParam | Play | Acquire | AcquireWeighed | AcquireTtl


class SetLatchEn(Model):
    enable: Value
    duration: Immediate


class LatchRst(Model):
    duration: Value


Trigger = SetLatchEn | LatchRst


class Wait(Model):
    duration: Value


class WaitTrigger(Model):
    trigger: Value
    duration: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.trigger) == type(self.duration)
        return self


class WaitSync(Model):
    duration: Value


WaitOps = Wait | WaitTrigger | WaitSync

RealTimeInstr = Conditional | Io | Trigger | WaitOps


Instruction = Q1Instr | RealTimeInstr


class Line(Model):
    label: str
    instruction: Instruction
    comment: str


class Block(Model):
    line: list[Line]
