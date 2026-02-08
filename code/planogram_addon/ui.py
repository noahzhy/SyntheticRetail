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
    horizontal_spacing: FloatProperty(name="组内空隙", default=0.02, min=0.0)
    segment_spacing: FloatProperty(name="组间空隙", default=0.05, min=0.0)
    edge_margin: FloatProperty(name="边距", default=0.05, min=0.0, description="Minimum distance from shelf edges")
    max_per_level: IntProperty(name="每层最大SKU数", default=50, min=1)
    align_mode: EnumProperty(
        name="对齐方式",
        items=[('LEFT', "Left", ""), ('CENTER', "Center", ""), ('RIGHT', "Right", "")]
    )
    fill_order: EnumProperty(
        name="填充顺序",
        items=[('TOP_DOWN', "Top → Bottom", ""), ('BOTTOM_UP', "Bottom → Top", "")]
    )
    random_seed: IntProperty(name="随机种子", default=42)
    allow_partial: BoolProperty(name="允许可部分填充", default=True)


class PLANOGRAM_PT_panel(bpy.types.Panel):
    # show version on panel 
    bl_label = "Retail Planogram"
    bl_idname = "PLANOGRAM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Planogram"

    def draw(self, context):
        props = context.scene.planogram_props
        layout = self.layout
        
        # 显示版本号
        from . import bl_info
        version_str = f"v{bl_info['version'][0]}.{bl_info['version'][1]}.{bl_info['version'][2]}"
        box = layout.box()
        box.label(text=f"版本: {version_str}", icon='INFO')
        
        layout.prop_search(props, "product_collection", bpy.data, "collections")
        layout.prop_search(props, "shelf_collection", bpy.data, "collections")
        layout.prop(props, "min_facing")
        layout.prop(props, "max_facing")
        layout.prop(props, "min_depth")
        layout.prop(props, "max_depth")
        layout.prop(props, "horizontal_spacing")
        layout.prop(props, "segment_spacing")
        layout.prop(props, "edge_margin")
        layout.prop(props, "max_per_level")
        layout.prop(props, "align_mode")
        layout.prop(props, "fill_order")
        layout.prop(props, "random_seed")
        layout.prop(props, "allow_partial")
        layout.operator("planogram.layout", text="生成陈列布局")
        layout.operator("planogram.clear", text="清空陈列布局", icon='TRASH')

class PLANOGRAM_OT_clear(bpy.types.Operator):
    bl_idname = "planogram.clear"
    bl_label = "Clear SM Collections"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collections_to_remove = [c for c in bpy.data.collections if c.name.startswith("SM_")]
        
        for col in collections_to_remove:
            # Unlink objects first to ensure safe removal if needed, 
            # but batch_remove on collection might be enough if we want to remove the collection data block. 
            # However, we want to remove the objects inside too if they are not used elsewhere.
            # User specifically mentioned batch_remove.
            # bpy.data.batch_remove(ids)
            
            # Use batch_remove for objects in these collections
            # Note: Objects might be shared. If we copied them, they are new. 
            # If we just linked existing ones, we shouldn't delete them. 
            # In placement.py, we do create_sku_instance -> source_obj.copy(). 
            # So they are new objects (linked to same data/mesh). 
            # Deleting the object is fine, keeping mesh is fine.
            
            objs_to_remove = [o for o in col.objects]
            if objs_to_remove:
                bpy.data.batch_remove(objs_to_remove)
                
        # Now remove the collections themselves
        if collections_to_remove:
            bpy.data.batch_remove(collections_to_remove)
            
        return {'FINISHED'}

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
    bpy.utils.register_class(PLANOGRAM_OT_clear)

def unregister():
    bpy.utils.unregister_class(PLANOGRAM_OT_clear)
    bpy.utils.unregister_class(PLANOGRAM_OT_layout)
    bpy.utils.unregister_class(PLANOGRAM_PT_panel)
    del bpy.types.Scene.planogram_props
    bpy.utils.unregister_class(PlanogramProperties)
