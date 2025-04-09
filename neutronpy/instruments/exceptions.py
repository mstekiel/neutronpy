"""Custom Errors for the instrument module.

DEV:
Every error is a subclass of `Warning`, so that it can be used either as a:
- warning: warn(SomeProblem('Some message'))
- exception: raise SomeProblem('Some message')

Idea here was to continue any calculations despite wrong input parameters.
This seems to be what numpy is doing and I like it.
"""
from warnings import warn

class DetectorError(Warning):
    """For Exceptions related to the properties of the Detector
    """
    pass


class ScatteringTriangleNotClosed(Warning):
    """For Exceptions related to the Scattering Triangle not closing
    """
    pass

class VectorNotInScatteringPlane(Warning):
    """For Exceptions related to vector not in scattering plane
    """
    pass

class MonochromatorError(Warning):
    """For Exceptions related to the Monochromator properties
    """
    pass


class AnalyzerError(Warning):
    """For Exceptions related to the Monochromator properties
    """
    pass

class ChopperError(Warning):
    """For Exceptions related to the Chopper properties
    """
    pass


class GoniometerError(Warning):
    """For Exceptions related to the Goniometer positions
    """
    pass


class InstrumentError(Warning):
    """For general, unclassified types of Instrument Exceptions
    """
    pass
