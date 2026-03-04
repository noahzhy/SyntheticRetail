## Plan: Blender 合成数据工厂（生产级）

在你当前“空仓库”的基础上，先落地一个可规模化的最小生产闭环：用 Blender/GN 负责稳定几何与通道渲染，用系统 Python 负责规则、分布控制、调度与 QC；所有随机来自可审计 seed，所有产物带 manifest 便于回滚/对账。计划会把“GN slot system 模板规范 + Python 规则引擎骨架 + bbox/polygon/occlusion + 分布监控”拆成可长期维护的模块边界与文件结构。

### Steps
1. 定义工程骨架与配置：新增 configs 与 src 包结构，确定 dataset 输出布局与 manifest 字段（参考 main.py 作为入口占位或替换为 CLI）。
2. 制作 GN Slot System 模板：在 Blender 文件中创建 Shelf→Slots 点阵、写入 slot attributes、提供 Python 可写入的 instance hook 属性与 debug 可视化规范。
3. 落地 SKU 与规则引擎骨架：实现 SKU catalog 读取、可复现 seed 生成、规则分层（slot/shelf/scene）与约束求解输出“场景配方”。
4. 实现 Blender 渲染与通道输出：提供 Blender 侧脚本（bpy）加载配方、写入 GN attributes、渲染 RGB + instance id + depth（用于 occlusion）。
5. 标注导出：基于 instance mask 计算 bbox、polygon（轮廓简化）、occlusion（visible pixel ratio），按你指定格式写 annotations 与 per-image metadata。
6. 分布监控与 QC：增量统计 SKU/category 分布、遮挡率、bbox 越界/空图/穿模等规则；异常丢弃并重采样，输出可审计 QC 报告。

### Further Considerations
1. 标注交付格式先对齐：自定义的标记格式 Pascal VOC 的。
2. 资产输入约定：SKU 模型是 .blend 资产库。
