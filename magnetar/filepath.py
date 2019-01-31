import os


def make_magprop_cwd(file, home_dir=None):
    """
Function to return an absolute file path to a specific folder.

    :param file: The file that this function is being called from
    :param home_dir: Optional directory to an absolute path for
    :return: An absolute file path to home_dir
    """
    # Get absolute file path of the parsed file
    here = os.path.abspath(os.path.dirname(file))

    # Split the file path by
    here_head, here_tail = os.path.split(here)

    if home_dir is None:
        home_dir = "magprop"

    # Check that the file path tail is the home directory we desire
    while here_tail != home_dir:
        here_head, here_tail = os.path.split(here_head)

    # Return home directory path
    return os.path.join(here_head, here_tail)
