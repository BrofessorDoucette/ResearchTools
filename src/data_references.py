import builtins
import datetime
import enum
import numpy as np
import numpy.typing as npt
import typing

class REPT(enum.Enum):
    
    FESA = 0
    L = 1
    MLT = 2
    EPOCH = 3
    ENERGIES = 4


class REPTDataRefContainer:
    
    '''
    This is just a class to package all of the REPT data references into one immutable tuple.
    '''
    
    def __init__(self, 
                fesa: npt.NDArray[np.float64],
                L: npt.NDArray[np.float64], 
                mlt: npt.NDArray[np.float64], 
                epoch: npt.NDArray[np.object_], 
                energies: npt.NDArray[np.float64]) -> None:

        #Tuples are immutable, which makes them nice for holding references.
        self._data = (fesa, L, mlt, epoch, energies)

    def get_all_data(self) -> tuple[npt.NDArray[np.float64],
                                    npt.NDArray[np.float64],
                                    npt.NDArray[np.float64],
                                    npt.NDArray[np.object_],
                                    npt.NDArray[np.float64]]:
        
        '''Returns all of the REPT Data in a tuple of NDArrays.
        This method is preferred over get_data as it preserves type hints for static analysis,
        However, it requires that the ordering of the return values is cautiously preserved,
        use get_data when you can't be bothered to get the ordering of the return values correct..'''
        
        
        return self._data
    
    def get_data(self, index: REPT) -> npt.NDArray[typing.Any]:
        
        if isinstance(index, builtins.int):
            
            return self._data[index]
        
        elif isinstance(index, REPT):
        
            return self._data[index.value]

class POES(enum.Enum):

    EPOCH = 0
    MEP_ELE_FLUX = 1
    L = 2
    MLT = 3
    NAIVE_CHORUS_AMPLITUDES = 4

class POESDataRefContainer:
    '''
    This is just a class to package all of the POES data references into one immutable tuple.
    '''

    def __init__(self,
                 epoch : npt.NDArray[np.object_],
                 mep_ele_flux : npt.NDArray[np.float32],
                 L : npt.NDArray[np.float32],
                 mlt : npt.NDArray[np.float32],
                 naive_chorus_amplitudes : npt.NDArray[np.float32],
                 satid: str) -> None:

        # Tuples are immutable, which makes them nice for holding references.
        self._data = (epoch, mep_ele_flux, L, mlt, naive_chorus_amplitudes)
        
        self.satid = satid

    def get_all_data(self) -> tuple[npt.NDArray[np.object_],
                                npt.NDArray[np.float32],
                                npt.NDArray[np.float32],
                                npt.NDArray[np.float32],
                                npt.NDArray[np.float32]]:

        '''Returns all of the REPT Data in a tuple of NDArrays.
        This method is preferred over get_data as it preserves type hints for static analysis,
        However, it requires that the ordering of the return values is cautiously preserved,
        use get_data when you can't be bothered to get the ordering of the return values correct..'''

        return self._data

    def get_data(self, index: POES) -> npt.NDArray[typing.Any]:

        if isinstance(index, builtins.int):

            return self._data[index]

        elif isinstance(index, POES):
            return self._data[index.value]
        