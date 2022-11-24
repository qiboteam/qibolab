How to connect Qibolab to your lab 
==================================

In this section we will see how to let ``Qibolab`` communicate with your lab's instruments and run an experiment.
``Qibolab`` has an abstract class ``AbstractInstrument`` with some abstract methods such as ``start``, ``stop``, ``connect``. 
In order to set up one instrument, you have to build a child class and implement the methods you need.
It follows a little example, where there are two classes (``DummyInstrument_1`` and ``DummyInstrument_2``) 
with a ``measure`` method that generates a random number between ``self.arg1`` and ``self.arg2``

.. code-block:: python

    class DummyInstrument_1(AbstractInstrument):

        def __init__(self,arg1,arg2):
            self.arg1 = arg1
            self.arg2 = arg2
            
        def start(self):
            pass 

        def stop (self):
            pass

        def connect(self):
            pass

        def disconnect(self):
            pass
        
        def setup(self):
            pass

        def measure(self):
            return np.random.uniform(self.arg1,self.arg2)

    class DummyInstrument_2(AbstractInstrument):

        def __init__(self,arg1,arg2):
            self.arg1 = arg1
            self.arg2 = arg2
            
        def start(self):
            pass 

        def stop (self):
            pass

        def measure(self):
            return np.random.uniform(self.arg1,self.arg2)
        
        def connect(self):
            pass

        def disconnect(self):
            pass
        
        def setup(self):
            pass

    

After all the devices have a proper class, they have to be coordinated to perform an experiment.
In ``Qibolab`` we can do this with a class (``DummyPlatform``) that inherits the methods from ``AbstractPlatform`` 
and reads the useful information from the runcard below (in this example we save it as ``dummy.yml``):

.. code-block:: yaml

    nqubits: 1
    description: Dummy platform runcard to use as example.

    qubits: [0]

    instruments: 
        instrument1:
            arg1: 0
            arg2: 1
        instrument2:
            arg1: 0
            arg2: 1

The class ``DummyPlatform`` has a method ``measure`` that calls the same method of 
the corresponding two instruments' classes and return a list of their outputs.

.. code-block:: python

    class DummyPlatform(AbstractPlatform):

        def __init__(self, name, runcard):
            self.name = name
            self.runcard = runcard
            with open(runcard) as file:
                self.settings = yaml.safe_load(file)
            arg1 = self.settings.get("instruments")["instrument1"]["arg1"]
            arg2 = self.settings.get("instruments")["instrument1"]["arg2"]
            self.instrument1 = DummyInstrument_1(arg1,arg2)
            arg1 = self.settings.get("instruments")["instrument2"]["arg1"]
            arg2 = self.settings.get("instruments")["instrument2"]["arg2"]
            self.instrument2 = DummyInstrument_2(arg1,arg2)

        def start(self):
            pass 

        def stop (self):
            pass

        def run_calibration(self, show_plots=False):  
            raise_error(NotImplementedError)
        
        def execute_pulse_sequence(self, sequence, nshots=None): 
            raise_error(NotImplementedError)

        def measure (self):
            return [self.instrument1.measure(), self.instrument2.measure()]

To start the experiment, simply initialize the platform and launch the desired method

.. code-block:: python

    platform = DummyPlatform("dummy","dummy.yml")
    output = platform.measure()



