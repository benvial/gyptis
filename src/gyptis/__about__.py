"""
Information about the package.
"""

try:
    # Python 3.8
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

try:
    data = metadata.metadata("gyptis")
    __version__ = metadata.version("gyptis")
    __author__ = data.get("author")
    __description__ = data.get("summary")
except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __description__ = "unknown"
