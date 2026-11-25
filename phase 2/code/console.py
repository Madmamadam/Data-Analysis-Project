from time import strftime


def print_status(*values: object, **print_options) -> None:
    """Print a status message with the current local time."""
    timestamp = f"[{strftime('%H:%M:%S')}]"
    if values:
        first, *remaining = values
        print(f"{timestamp} {first!s}", *remaining, **print_options)
    else:
        print(timestamp, **print_options)
