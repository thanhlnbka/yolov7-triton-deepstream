import platform
import sys


def is_aarch64():
    return platform.uname()[4] == 'aarch64'

sys.path.append('/opt/nvidia/deepstream-6.1/deepstream/lib')