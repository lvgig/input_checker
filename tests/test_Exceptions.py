from input_checker.exceptions import InputCheckerError


def test_inheritance():
    """Test that InputCheckerError inherits from Exception."""

    checker_error = InputCheckerError("aaaa")

    assert isinstance(
        checker_error, Exception
    ), "InputCheckerError is not inheriting from Exception"


def test_exception_args_set(mocker):
    """Test that the positional args passed to InputCheckerError are set on the object."""

    pos_args = ("aaaa", 1234, "bbbb")

    checker_error = InputCheckerError(*pos_args)

    # this is effectively testing Exception.__init__ is called correctly
    # as all positional args passed are set in the args attribute
    assert (
        checker_error.args == pos_args
    ), "args attribute not set correctly on InputCheckerError"
