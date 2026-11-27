"""Echo: expressive voice dubbing that keeps each character's own voice."""

__version__ = "0.1.0"


class EchoError(RuntimeError):
    """A step of the dub could not run; the message says what to do."""
