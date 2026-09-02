"""Shared declared-channel helpers, reused by every needs>={"declared"} method below."""


def declared(view):
    """Collect D from the registry once at build: n messages, per the message-accounting rule."""
    view.ledger.message(view.n)
    return view.declared


def scan(view, D, f):
    """A flat read of one family's column, charged as an O(n) comparison scan."""
    view.ledger.compare(view.n)
    return D[:, f]
