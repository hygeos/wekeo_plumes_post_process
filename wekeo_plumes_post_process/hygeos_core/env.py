from dotenv import load_dotenv, find_dotenv
from os import environ
from pathlib import Path
from typing import Optional

load_dotenv(dotenv_path=find_dotenv(usecwd=True)) 


def getvar(
    envvar: str,
    default: str|None = None,
    ) -> str:
    """
    Returns the value of environment variable `envvar`. If this variable is not defined, returns default.

    The environment variable can be defined in the users `.bashrc`, or in a file `.env`
    in the current working directory.

    Args:
        envvar: the input environment variable
        default: the default return, if the environment variable is not defined
    
    Returns:
        the requested environment variable or the default if the var is not defined and a default has been provided.
    """
    variable = None
    if envvar in environ:
        variable = environ[envvar]
        
    elif default is None:
        raise KeyError(f"{envvar} is not defined, and no default has been provided.")
    else:
        variable = default

    return variable

def getdir(
    envvar: str,
    default: Optional[Path] = None,
    create: Optional[bool] = None,
) -> Path:
    """
    Returns the value of environment variable `envvar`, assumed to represent a
    directory path. If this variable is not defined, returns default.

    The environment variable can be defined in the users `.bashrc`, or in a file `.env`
    in the current working directory.

    Args:
        envvar: the input environment variable
        default: the default path, if the environment variable is not defined
            default values are predefined for the following variables:
                - DIR_DATA : "data" (in current working directory)
                - DIR_STATIC : DIR_DATA/"static"
                - DIR_SAMPLES : DIR_DATA/"sample_products"
                - DIR_ANCILLARY : DIR_DATA/"ancillary"
        create: whether to silently create the directory if it does not exist.
            If not provided this parameter defaults to False except for DIR_STATIC,
            DIR_SAMPLES and DIR_ANCILLARY.
    
    Returns:
        the path to the directory.
    """
    use_default = envvar not in environ
    default_create = False
    if envvar in environ:
        dirname = Path(environ[envvar])
    else:
        if default is None:
            # use a predefined default value
            if envvar == 'DIR_DATA':
                # Root of the data directory
                # All data in this directory are assumed disposable, and should be
                # downloaded on the fly
                # defaults to 'data' in the current working directory
                dirname = Path('data')
            elif envvar == 'DIR_STATIC':
                # static data files, required for processing
                dirname = getdir('DIR_DATA')/"static"
                default_create = True
            elif envvar == 'DIR_SAMPLES':
                # sample products, used for testing
                dirname = getdir('DIR_DATA')/"sample_products"
                default_create = True
            elif envvar == 'DIR_ANCILLARY':
                # ancillary data (downloaded on the fly)
                dirname = getdir('DIR_DATA')/"ancillary"
                default_create = True
            else:
                raise KeyError(f"{envvar} is not defined, and no default has been "
                               "provided.")
        else:
            dirname = Path(default)

    if create is None:
        create = default_create
    
    if not dirname.exists():
        if create:
            dirname.mkdir(exist_ok=True)
        else:
            raise NotADirectoryError(
                (f"Environment variable '{envvar}' is undefined, using default "
                 f"value '{dirname}'. " if use_default else "") + 
                f"Directory '{dirname}' does not exist. You may want to initialize it "
                f"with the following command: 'mkdir {dirname}'")

    return dirname