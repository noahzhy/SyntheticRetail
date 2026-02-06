import bpy
from bpy.props import PointerProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty, StringProperty
from . import placement

class PlanogramProperties(bpy.types.PropertyGroup):
    product_collection: StringProperty(name="Product Collection")
    shelf_collection: StringProperty(name="Shelf Collection")
    min_facing: IntProperty(name="Min Facing", default=1, min=1)
    max_facing: IntProperty(name="Max Facing", default=5, min=1)
    min_depth: IntProperty(name="Min Depth", default=1, min=1)
    max_depth: IntProperty(name="Max Depth", default=3, min=1)
    horizontal_spacing: FloatProperty(name="Horizontal Spacing", default=0.02, min=0.0)
    segment_spacing: FloatProperty(name="Segment Spacing", default=0.05, min=0.0)
    max_per_level: IntProperty(name="Max Products/Level", default=20, min=1)
    align_mode: EnumProperty(
        name="Alignment",
        items=[('LEFT', "Left", ""), ('CENTER', "Center", ""), ('RIGHT', "Right", "")]
    )
    fill_order: EnumProperty(
        name="Fill Order",
        items=[('TOP_DOWN', "Top → Bottom", ""), ('BOTTOM_UP', "Bottom → Top", "")]
    )
    random_seed: IntProperty(name="Random Seed", default=42)
    allow_partial: BoolProperty(name="Allow Partial Placement", default=True)

class PLANOGRAM_PT_panel(bpy.types.Panel):
    bl_label = "Retail Planogram"
    bl_idname = "PLANOGRAM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Planogram"

    def draw(self, context):
        props = context.scene.planogram_props
        layout = self.layout
        layout.prop_search(props, "product_collection", bpy.data, "collections")
        layout.prop_search(props, "shelf_collection", bpy.data, "collections")
        layout.prop(props, "min_facing")
        layout.prop(props, "max_facing")
        layout.prop(props, "min_depth")
        layout.prop(props, "max_depth")
        layout.prop(props, "horizontal_spacing")
        layout.prop(props, "segment_spacing")
        layout.prop(props, "max_per_level")
        layout.prop(props, "align_mode")
        layout.prop(props, "fill_order")
        layout.prop(props, "random_seed")
        layout.prop(props, "allow_partial")
        layout.operator("planogram.layout", text="Generate Planogram")

class PLANOGRAM_OT_layout(bpy.types.Operator):
    bl_idname = "planogram.layout"
    bl_label = "Generate Planogram"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.planogram_props
        placement.generate_planogram(context, props)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(PlanogramProperties)
    bpy.types.Scene.planogram_props = PointerProperty(type=PlanogramProperties)
    bpy.utils.register_class(PLANOGRAM_PT_panel)
    bpy.utils.register_class(PLANOGRAM_OT_layout)

def unregister():
    bpy.utils.unregister_class(PLANOGRAM_OT_layout)
    bpy.utils.unregister_class(PLANOGRAM_PT_panel)
    del bpy.types.Scene.planogram_props
    bpy.utils.unregister_class(PlanogramProperties)
