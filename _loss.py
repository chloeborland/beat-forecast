# _loss.py
"""
Compatibility shim for loading a legacy joblib artifact that references a module named `_loss`.

Your saved model was pickled with references like `_loss.CyHalfBinomialLoss`.
On machines/environments where that internal class/module path doesn't exist, joblib.load fails.

This shim defines the expected symbols so unpickling can succeed.
Best long-term fix: re-export the model artifact from a stable environment (Python 3.11 + pinned sklearn).
"""

class _NoOp:
    """Generic placeholder object used as a stand-in for missing loss functions/classes."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


# The specific attribute your pickle is requesting
class CyHalfBinomialLoss(_NoOp):
    pass


def __getattr__(name: str):
    """
    Called when an attribute is not found on this module (Python 3.7+).
    Return a placeholder so unpickling can resolve references like `_loss.SomeName`.
    """
    return _NoOp