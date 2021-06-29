import urllib3

http = urllib3.PoolManager()

class MCAttenuator:
    def __init__(self, address: str):
        self.address = address

    def set_attenuation(self, attenuation: int) -> None:
        http.request('GET', 'http://{}/SETATT={}'.format(self.address, attenuation))
