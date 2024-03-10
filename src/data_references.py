import builtins
import enum

class DataRef(enum.Enum):
    
    FESA = 0
    L = 1
    MLT = 2
    EPOCH = 3
    ENERGIES = 4

class DataRefContainer:
    
    '''
    This is just a class to package all of the REPT data references into one immutable tuple.
    '''
    
    def __init__(self, FESA, L, MLT, EPOCH, ENERGIES) -> None:
        
        #Tuples are immutable, which makes them nice for holding references.
        self._data = (FESA, L, MLT, EPOCH, ENERGIES)
        
        
    def get_all_data(self):
                
        return self._data
    
    def get_data(self, index: DataRef):
        
        if isinstance(index, builtins.int):
            
            return self._data[index]
        
        elif isinstance(index, DataRef):
        
            return self._data[index.value]