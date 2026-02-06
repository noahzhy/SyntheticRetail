def report_error(msg):
    import bpy
    bpy.ops.message.error('INVOKE_DEFAULT', message=msg)

def segment_skus(skus, props):
    # 每个SKU为一个segment
    return [(sku, {}) for sku in skus]


def create_sku_instance(source_obj, target_collection):
    new_obj = source_obj.copy()
    if source_obj.data:
        new_obj.data = source_obj.data
    target_collection.objects.link(new_obj)
    return new_obj


def register():
    pass


def unregister():
    pass
