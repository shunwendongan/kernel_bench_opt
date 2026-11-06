# README 手绘图

本目录的五张图使用原生 SVG 绘制，文字、线条、图标和颜色均可编辑。近白纸面、铅笔式轮廓与蓝绿珊瑚紫配色用于解释系统关系；它们是概念图，不是实测性能图。

| 文件 | 内容 |
|---|---|
| [01-workflow-overview.svg](01-workflow-overview.svg) | GitHub、本地主机、端侧容器、分析与评测回路 |
| [02-task-sandbox.svg](02-task-sandbox.svg) | 只读共享资产、Task / Candidate / Run、独立输出与待接入的设备排队 |
| [03-hotspot-to-kernel.svg](03-hotspot-to-kernel.svg) | profile热点、来源图候选、引擎分派与实际kernel证据 |
| [04-benchmark-gates.svg](04-benchmark-gates.svg) | L1算子、L2模块/子图、L3完整请求与结果归档 |
| [05-knowledge-cycle.svg](05-knowledge-cycle.svg) | 原始证据到知识资产，本地大文件与Git精选证据的身份关联 |

图中关系以 [工作流程说明](../SOP.md)、[容器指南](../CONTAINERS.md) 及既有实验记录为依据；设计时参考了两份本地历史HTML的模块协同、来源映射和多级验证内容，原HTML未复制入库。

画布宽1280像素；中文字体优先楷体，回退到系统中文字体。文件没有脚本、远程图片或外部字体请求，含有可读的title/desc。修改后可用浏览器或本机SVG工具预览。平台尚未接通的环节继续标为目标/待接入，不能只改图让它看起来已经运行。
