#This file will just import the config and hopefully let us clean up a lot of code everywhere eventually
import yaml
import os


def replace_all_keys_in_string_with_values(string : str, map : dict = {}):
    '''Replaces all occurances of the keys in the string with the values. 
       This is really the backbone of what makes the download_from_global_config method versatile. 
       Its essentially just a wrapper of a recursive wget call, but a useful one nonetheless.'''
    
    if not map:
        return string
    
    for key in map.keys():
        string = string.replace(key, str(map[key]))
        
    return string

class Config:
    
    
    def __init__(self, path : str = "") -> None:
        
        '''Set the config path here, either relative to where python is executed, or an absolute path.'''
        
        if not path:
        
            self._config_path = r"C:\Dev\Research\REPT_Enhancements_Tool\config.yaml"
            
        else:
            
            self._config_path = path

    
    def load(self) -> tuple[dict, str]:
        
        '''Returns the config dictionary, and the absolute path to the config if needed.'''
        
        config_path = os.path.abspath(self._config_path)
        
        with open(config_path, "r") as config_file:
            
            config = yaml.safe_load(config_file)    
            
        return config, config_path
            