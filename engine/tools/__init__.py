"""Tool registry for the Curiosity Engine.

Tools are Python classes that subclass `Tool`. Subclassing auto-registers them with
the module-level `registry`. Modules in this package are imported lazily by
`discover_tools()` which walks this directory on first call.

Tools are sync. They take a dict of validated args and return a string.
"""

from engine.tools.base import (
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    discover_tools,
    registry,
)

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "discover_tools",
    "registry",
]
