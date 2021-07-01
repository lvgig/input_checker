class InputCheckerError(Exception):
    """Exception with custom name, does not add any extra functionality."""

    def __init__(self, *args):

        super().__init__(*args)
