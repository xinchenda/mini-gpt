"""Handle AutoDL shutdown files lacking an executable script header.

The platform timer and a console status check remain independent safeguards.
"""
import errno
import subprocess


def power_off(timeout=30):
    command='/usr/bin/shutdown'
    try:
        return subprocess.run([command],check=True,timeout=timeout)
    except OSError as error:
        if error.errno != errno.ENOEXEC:
            raise
        return subprocess.run(['/bin/bash',command],check=True,timeout=timeout)
