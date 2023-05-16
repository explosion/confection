DEFAULT_FROZEN_DICT_ERROR = (
    "Can't write to frozen dictionary. This is likely an internal "
    "error. Are you writing to a default function argument?"
)

DEFAULT_FROZEN_LIST_ERROR = (
    "Can't write to frozen list. Maybe you're trying to modify a computed "
    "property or default function argument?"
)


class SimpleFrozenDict(dict):
    """Simplified implementation of a frozen dict, mainly used as default
    function or method argument (for arguments that should default to empty
    dictionary). Will raise an error if the user attempts to add to dict.
    """

    def __init__(
        self,
        *args,
        error: str = DEFAULT_FROZEN_DICT_ERROR,
        **kwargs,
    ) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):
        raise NotImplementedError(self.error)

    def update(self, other):
        raise NotImplementedError(self.error)


class SimpleFrozenList(list):
    """Wrapper class around a list that lets us raise custom errors if certain
    attributes/methods are accessed. Mostly used for properties that return an
    immutable list (and that we don't want to convert to a tuple to not break
    too much backwards compatibility). If a user accidentally calls
    frozen_list.append(), we can raise a more helpful error.
    """

    def __init__(
        self,
        *args,
        error: str = DEFAULT_FROZEN_LIST_ERROR,
    ) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):
        raise NotImplementedError(self.error)
