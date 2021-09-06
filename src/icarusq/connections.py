from abc import ABC, abstractmethod
from qibo.config import raise_error


class Connection(ABC):

    @abstractmethod
    def exec_command(self, command): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def __enter__(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def __exit__(self, *args): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def putfo(self): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def getfo(self): # pragma: no cover
        raise_error(NotImplementedError)


class ParamikoSSH(Connection):

    def __init__(self, hostname, username, password):
        import paramiko # pylint: disable=E0401
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=hostname, username=username, password=password)

        self.sftp = None
        self.sftp_counter = 1
        self.putfo_dir = '/tmp/wave_ch{}.csv'
        self.getfo_dir = '/tmp/ADC_CH{}.txt'

    def exec_command(self, command):
        self.ssh.exec_command(command)

    def __enter__(self):
        self.sftp = self.ssh.open_sftp()
        self.sftp_counter = 1
        return self

    def __exit__(self, *args):
        self.sftp.close()

    def putfo(self, dump):
        self.sftp.putfo(dump, self.putfo_dir.format(self.sftp_counter))
        self.sftp_counter += 1

    def getfo(self, dump):
        self.sftp.getfo(self.getfo_dir.format(self.sftp_counter), dump)
        self.sftp_counter += 1
        return dump
