import sys
from io import TextIOWrapper, BytesIO

class RedirectPrint:
    def __init__(self):
        pass

    def start(self):
        # setup the environment
        self.old_stdout = sys.stdout
        sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)

    def stop(self):
        sys.stdout.close()
        sys.stdout = self.old_stdout

    def get_output(self):
        # get output
        sys.stdout.seek(0)      # jump to the start
        out = sys.stdout.read() # read output

        return out