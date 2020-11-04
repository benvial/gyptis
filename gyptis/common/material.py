__all__ = ["Material"]


class Material(object):
    """
    This class defines a material with electromagnetic properties.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a material.
        """
        for material in args:
            for key, value in material.__dict__.items():
                setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    # @staticmethod
    # def py():
    #   """
    #   Return an instance of Material with the material properties
    #   similar to Permalloy."""
    #   return Material(alpha=0.02, Aex=13e-12, ms=8e5)
