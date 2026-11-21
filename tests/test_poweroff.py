import errno
import unittest
from unittest.mock import patch
from scripts.autodl_poweroff import power_off


class Poweroff(unittest.TestCase):
    @patch('scripts.autodl_poweroff.subprocess.run')
    def test_native_command(self,run):
        power_off()
        run.assert_called_once_with(['/usr/bin/shutdown'],check=True,timeout=30)

    @patch('scripts.autodl_poweroff.subprocess.run')
    def test_headerless_script_falls_back_to_bash(self,run):
        run.side_effect=[OSError(errno.ENOEXEC,'Exec format error'),object()]
        power_off()
        self.assertEqual(run.call_args_list[1].args[0],['/bin/bash','/usr/bin/shutdown'])

    @patch('scripts.autodl_poweroff.subprocess.run')
    def test_other_errors_propagate(self,run):
        run.side_effect=PermissionError(errno.EACCES,'Permission denied')
        with self.assertRaises(PermissionError):power_off()
        self.assertEqual(run.call_count,1)
