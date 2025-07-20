import re


def to_class_name(s: str) -> str:
    """
    Convert a string to a valid Python class name.

    Args:
        s: The string to convert.

    Returns:
        A valid Python class name.
    """
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)  # Replace invalid characters with underscores
    if not s[0].isalpha():  # Prepend an underscore if the first character isn't a letter
        s = "_" + s
    return s.title()


def patch_class_name(object: object, name: str):
    """
    Patch the class name of an object.
    This is useful when you want to create a class with a dynamic name.

    Args:
        object: The object to patch.
        name: The new class name.
    """
    object.__class__ = type(to_class_name(name), (object.__class__,), {})
