class ComponentTemplate(object):
    """General object class encompassing only properties of the device,
     without further functionalities. E.g. source, monitor, detector."""

    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    def __repr__(self):
        return "{1}({0})".format(', '.join(
            ['{0}={1}'.format(key, getattr(self, key)) for key in self.__dict__.keys() if
             getattr(self, key, None) is not None and key != 'name']), self.name)