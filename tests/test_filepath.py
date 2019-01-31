import os
from magnetar.filepath import make_magprop_cwd


def test_magnetar_filepath():

    exp_tail = "magprop"

    home_dir = make_magprop_cwd(__file__)
    head, tail = os.path.split(home_dir)

    assert tail == exp_tail
