import bpy
import math

def auto_sort_objects_grid():
    # 获取当前选中的物体
    selected_objs = bpy.context.selected_objects
    
    if not selected_objs:
        print("没有选中任何物体！")
        return
        
    n_objs = len(selected_objs)
    if n_objs <= 1:
        print("选中的物体不足2个，无需重新排版。")
        return
        
    # 1. 计算所有选中物体在X和Z方向的最大和最小边界，以及平均Y坐标进行对齐
    min_x = min(obj.location.x for obj in selected_objs)
    max_x = max(obj.location.x for obj in selected_objs)
    min_z = min(obj.location.z for obj in selected_objs)
    max_z = max(obj.location.z for obj in selected_objs)
    avg_y = sum(obj.location.y for obj in selected_objs) / n_objs
    
    # 2. 对物体按照 Z坐标递减（从高到低），然后X坐标递增（从左到右） 的顺序进行排序
    # 注：通过设置 -obj.location.z 实现从上到下排列，也可以根据需要调整排序规则
    sorted_objs = sorted(selected_objs, key=lambda obj: (-obj.location.z, obj.location.x))
    
    # 3. 计算网格的行数和列数（根据总数开方估算，使其尽量接近接近正方形的矩形排列）
    cols = math.ceil(math.sqrt(n_objs))
    rows = math.ceil(n_objs / cols)
    
    # 4. 计算网格的步长/间距
    spacing_x = (max_x - min_x) / (cols - 1) if cols > 1 else 0
    spacing_z = (max_z - min_z) / (rows - 1) if rows > 1 else 0
    
    # 5. 遍历排序好的物体并重新为其赋予网格坐标
    for i, obj in enumerate(sorted_objs):
        row = i // cols
        col = i % cols
        
        # 根据行列索引分别计算新的位置
        new_x = min_x + col * spacing_x
        # Z轴方向从高处往低处摆放
        new_z = max_z - row * spacing_z
        
        # 更新物体的坐标，修改 X 和 Z，并将 Y 轴进行统一对齐
        obj.location.x = new_x
        obj.location.y = avg_y
        obj.location.z = new_z
        
    print(f"成功将 {n_objs} 个选中的物体整理成了 {rows} 行 {cols} 列的网格！边界范围 -> X: [{min_x:.2f}, {max_x:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]")

if __name__ == "__main__":
    auto_sort_objects_grid()
