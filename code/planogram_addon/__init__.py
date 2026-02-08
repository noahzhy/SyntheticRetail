bl_info = {
    "name": "Retail Planogram Automation",
    "author": "Noah Zhang",
    "version": (1, 0, 5),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Planogram",
    "description": "Automated multi-level, multi-segment retail shelf planogram layout",
    "category": "Object",
}

version = ".".join(map(str, bl_info["version"]))
ADDON_VERSION = version

import importlib
from . import ui, placement, geometry, collision, utils

modules = [ui, placement, geometry, collision, utils]

def register():
    for m in modules:
        if hasattr(m, 'register'):
            m.register()

def unregister():
    for m in reversed(modules):
        m.unregister()
