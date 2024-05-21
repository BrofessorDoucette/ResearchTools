import os


def verify_output_dir_exists(directory: str, force_creation: bool = False, hint: str = "") -> None:
    '''Check if the folder exists. If it doesn't, make the folder if force_creation is True.'''
    
    if not os.path.isdir(directory):
        
        if force_creation:
            
            os.makedirs(directory)
            
        else:
            if len(hint) > 0:
                
                raise Exception(f"\nHint: {hint}\nOutput dir: {directory} doesn't exist, and make_dirs flag is set to false! \nPlease make the directory or give me permission to make it for you by setting the make_dirs flag.\n")

            else:
                
                raise Exception(f"\nOutput dir: {directory} doesn't exist, and make_dirs flag is set to false! \nPlease make the directory or give me permission to make it for you by setting the make_dirs flag.\n")

            
def verify_input_dir_exists(directory: str, hint: str = "") -> None:
    
    if not os.path.isdir(directory):
        
        if len(hint) > 0:
            
            raise Exception(f"\nHint: {hint}\nInput dir: {directory} doesn't exist. \nCan't read required data.\n")
        
        else:
            
            raise Exception(f"\nInput dir: {directory} doesn't exist. \nCan't read required data.\n")
