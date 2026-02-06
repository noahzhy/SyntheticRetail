bl_info = {
    "name": "Retail Planogram Automation",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Planogram",
    "description": "Automated multi-level, multi-segment retail shelf planogram layout",
    "category": "Object",
}

import importlib
from . import ui, placement, geometry, collision, utils

modules = [ui, placement, geometry, collision, utils]

def register():
    for m in modules:
        importlib.reload(m)
        m.register()

def unregister():
    for m in reversed(modules):
        m.unregister()
