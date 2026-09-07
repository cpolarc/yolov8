# 手写 YOLOv8 实现计划

## 目标

使用 PyTorch 手写一个可训练、可验证、可推理的 YOLOv8 目标检测模型。

第一版建议固定为：

- 模型：YOLOv8n
- 任务：目标检测
- 输入尺寸：640 × 640
- 数据格式：YOLO TXT
- 深度学习框架：PyTorch
- 输出尺度：P3、P4、P5，对应 stride 8、16、32
- 检测方式：Anchor-Free
- 边界框回归：DFL（Distribution Focal Loss）
- 标签分配：Task-Aligned Assigner
- 后处理：置信度过滤 + NMS
- 训练方式：先实现单 GPU

第一版暂不实现：

- 实例分割
- 姿态估计
- OBB 旋转框
- 多目标跟踪
- ONNX、TensorRT 等模型导出
- DDP 多 GPU 训练
- 自动超参数搜索
- 完整的日志和回调系统

---

## 一、建议的代码目录

模型相关代码可以放在 `nn/` 目录中。其中基础组件放在 `nn/modules/`，完整模型结构放在 `nn/models/`。

```text
tmp_handmake/
├── README.md
├── nn/
│   ├── modules/
│   │   ├── conv.py
│   │   ├── block.py
│   │   ├── head.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── neck.py
│   │   ├── yolov8.py
│   │   └── __init__.py
│   └── __init__.py
├── data/
│   ├── dataset.py
│   ├── augment.py
│   └── dataloader.py
├── loss/
│   ├── assigner.py
│   ├── dfl_loss.py
│   └── detection_loss.py
├── engine/
│   ├── trainer.py
│   ├── validator.py
│   └── predictor.py
├── utils/
│   ├── boxes.py
│   ├── anchors.py
│   ├── nms.py
│   ├── metrics.py
│   └── checkpoints.py
├── configs/
│   ├── model.yaml
│   ├── data.yaml
│   └── train.yaml
└── tests/
    ├── test_model_shapes.py
    ├── test_boxes.py
    ├── test_dfl.py
    ├── test_assigner.py
    ├── test_loss.py
    ├── test_dataset.py
    └── test_inference.py
```

目录不必一开始全部创建。建议按照实现顺序逐步添加。

---

## 二、基础网络组件

首先实现 YOLOv8 使用的基础网络模块。

### 1. Conv 模块

需要包括：

- `Conv2d`
- `BatchNorm2d`
- `SiLU`
- 自动计算 padding
- 普通卷积
- `stride=2` 的下采样卷积
- 推理阶段的 Conv 和 BN 融合可以后期再实现

### 2. Bottleneck

需要包括：

- 两层卷积
- 可配置的残差连接
- 输入和输出通道一致时启用 shortcut

### 3. C2f

这是 YOLOv8 的主要特征提取模块，需要实现：

- 输入特征的通道变换与拆分
- 多个 Bottleneck 串联
- 保存各分支输出
- 多分支特征拼接
- 最后的通道融合

### 4. SPPF

SPPF 用于扩大感受野，需要实现：

- 通道压缩
- 连续三次最大池化
- 不同感受野特征的拼接
- 输出通道融合

### 5. Concat

用于 Neck 中不同来源特征的通道拼接。

### 6. Upsample

使用最近邻插值完成特征图上采样。

---

## 三、YOLOv8 Backbone

Backbone 负责从输入图片中提取多尺度特征。

需要按照 YOLOv8n 的深度和宽度配置完成以下结构：

1. 使用两次 `stride=2` 卷积进行初步下采样。
2. 使用 C2f 提取特征。
3. 继续逐级下采样。
4. 在不同阶段保留 P3、P4、P5 特征。
5. 在最深层连接 SPPF。

需要重点确认：

- 每层输入和输出通道数
- 每层特征图尺寸
- C2f 中 Bottleneck 的重复次数
- width multiplier
- depth multiplier
- 通道数的取整和上限规则

### Backbone 验收条件

输入形状：

```text
[B, 3, 640, 640]
```

输出三个尺度的特征：

- P3：约 80 × 80，stride=8
- P4：约 40 × 40，stride=16
- P5：约 20 × 20，stride=32

---

## 四、YOLOv8 Neck

YOLOv8 使用类似 PAN-FPN 的双向特征融合结构。

### 1. 自顶向下融合

1. P5 上采样。
2. 与 Backbone 输出的 P4 拼接。
3. 经过 C2f 融合。
4. 再次上采样。
5. 与 Backbone 输出的 P3 拼接。
6. 再经过 C2f 融合。

### 2. 自底向上融合

1. P3 融合结果通过卷积下采样。
2. 与中间的 P4 特征拼接。
3. 经过 C2f 融合。
4. P4 融合结果再次下采样。
5. 与 P5 特征拼接。
6. 经过 C2f 融合。

最终输出三个检测尺度：

- 小目标检测特征
- 中目标检测特征
- 大目标检测特征

---

## 五、Anchor-Free Detect Head

YOLOv8 的检测头需要在每个尺度上建立两个独立分支。

### 1. 分类分支

分类分支输出：

- 每个特征点对应的类别 logits
- 输出通道数等于类别数量 `nc`

YOLOv8 不单独预测传统的 objectness，最终置信度来自分类分支。

### 2. 边界框回归分支

每个位置预测相对于 anchor point 的四个距离：

- left
- top
- right
- bottom

每个距离不是直接预测一个连续值，而是预测 `reg_max` 个离散分布值。

常见设置：

```text
reg_max = 16
回归输出通道数 = 4 × reg_max
```

### 3. Detect Head 的其他职责

- 保存三个尺度的 stride：8、16、32
- 根据特征图生成 anchor points，也就是网格中心点
- 将 DFL 分布转换成四个连续距离
- 将距离转换成 `xyxy` 或 `xywh` 边界框
- 训练阶段返回原始分类和回归张量
- 推理阶段返回解码后的预测结果
- 初始化分类和回归分支的 bias

---

## 六、边界框工具

需要单独实现并测试一套边界框工具，包括：

- `xywh` 转 `xyxy`
- `xyxy` 转 `xywh`
- 归一化坐标转像素坐标
- 像素坐标转归一化坐标
- box clipping
- box area
- pairwise IoU
- GIoU
- DIoU
- CIoU
- anchor point 生成
- `bbox2dist`
- `dist2bbox`
- LetterBox 前后的坐标映射

这部分必须编写单元测试，因为训练中的很多问题都来自坐标格式、缩放或 padding 处理错误。

---

## 七、DFL 回归

YOLOv8 使用 DFL 表达边界框四个方向的距离。

### 1. DFL 解码

将每个方向的 `reg_max` 个 logits 转换为连续距离：

1. 对 logits 做 softmax。
2. 与 `[0, 1, ..., reg_max - 1]` 相乘。
3. 对结果求和，得到分布期望。
4. 输出连续距离。

### 2. DFL 训练损失

真实距离通常位于两个整数 bin 之间，需要：

1. 找到左侧整数 bin。
2. 找到右侧整数 bin。
3. 根据真实距离计算左右权重。
4. 分别计算两个 bin 的交叉熵。
5. 将两部分交叉熵加权求和。

需要特别处理：

- 将真实距离裁剪到 `reg_max - 1` 范围内
- 浮点边界
- 没有正样本的情况

---

## 八、Task-Aligned Assigner

Task-Aligned Assigner 负责把真实目标分配给特征图上的预测点，是训练部分的核心模块之一。

需要实现以下步骤：

1. 找出位于真实框内部的 anchor points。
2. 获取每个点对真实类别的预测分数。
3. 计算预测框与真实框之间的 IoU。
4. 根据分类分数和 IoU 计算 alignment metric。
5. 为每个真实框选择 top-k 候选点。
6. 解决同一个预测点匹配多个真实框的冲突。
7. 生成训练目标。

需要输出：

- foreground mask
- target boxes
- target labels
- target scores
- matched ground-truth index

必须处理的边界情况：

- 一张图片没有标注
- 某个真实框没有合格候选点
- 一个点同时被多个真实框选中
- 小目标只覆盖极少的特征点
- padding 后的无效标签
- 一个 batch 中每张图片的真实框数量不同

---

## 九、损失函数

YOLOv8 检测损失至少包含三部分。

### 1. 分类损失

- 使用 `BCEWithLogitsLoss`
- 分类目标不是简单的 0/1 one-hot
- 使用 Assigner 产生的质量加权 target score
- 根据正样本 target score 进行归一化

### 2. IoU 回归损失

- 只对正样本计算
- 推荐使用 CIoU
- 使用 target score 作为样本权重

### 3. DFL 损失

- 将真实框转换为 anchor point 到四条边的距离
- 只对正样本计算 Distribution Focal Loss
- 使用 target score 加权

最终损失由以下部分组成：

```text
总损失 = box loss + cls loss + dfl loss
```

三个损失项的 gain 应放入训练配置中，不要分散硬编码在实现里。

### 损失函数验收条件

- 随机输入时 loss 是有限值
- 空标签 batch 不崩溃
- loss 可以正常 backward
- 梯度中不出现 NaN 或 Inf
- 模型可以在少量图片上明显过拟合

---

## 十、数据集与 DataLoader

### 1. 数据读取

需要支持：

- 图片目录
- YOLO TXT 标签
- 每行格式为 `class x_center y_center width height`
- 归一化坐标
- 缺失标签文件
- 空标签文件
- 损坏图片检查
- 类别编号范围检查

每个样本至少应返回：

- image tensor
- class labels
- boxes
- batch index
- 原始图片尺寸
- 缩放比例和 padding 信息
- 图片路径

### 2. Collate Function

由于每张图片中的目标数量不同，需要自定义 batch 拼接：

- 图片 tensor 正常堆叠
- 所有标签统一拼接
- 为每个目标记录其所属的 batch index

### 3. 基础预处理

需要实现：

- LetterBox resize
- BGR/RGB 转换
- HWC 转 CHW
- `uint8` 转浮点数
- 除以 255 完成归一化
- batch 维度处理

---

## 十一、数据增强

建议分两个阶段实现。

### 第一阶段：保证训练流程正确

只实现：

- LetterBox
- 随机水平翻转
- HSV 增强

### 第二阶段：接近 YOLOv8 的训练效果

增加：

- Mosaic
- 随机仿射
- 平移
- 缩放
- 剪切
- MixUp（可选）
- 训练最后若干 epoch 关闭 Mosaic
- 多尺度训练（可选）

每次几何增强后都必须同步更新边界框，并删除：

- 面积过小的框
- 宽高无效的框
- 严重被裁剪的框

不建议一开始就实现 Mosaic，因为它会显著增加坐标排错难度。

---

## 十二、训练引擎

最小训练循环需要包含：

1. 读取配置。
2. 构建模型。
3. 创建 Dataset 和 DataLoader。
4. 创建优化器。
5. 执行前向传播。
6. 计算 loss。
7. 执行 AMP 反向传播。
8. 执行 optimizer step。
9. 更新学习率。
10. 更新 EMA。
11. 保存 checkpoint。
12. 每个 epoch 结束后验证。

训练引擎后续逐步支持：

- SGD
- AdamW
- warmup
- cosine learning rate
- weight decay 参数分组
- gradient accumulation
- AMP
- gradient clipping（可选）
- EMA
- checkpoint
- resume
- 随机种子
- 训练指标记录
- best 和 last 权重保存

第一版只实现单 GPU。只有单 GPU 训练完全正确后，再考虑 DDP。

---

## 十三、推理与后处理

推理流程需要包含：

1. 读取图片或视频帧。
2. 执行 LetterBox。
3. 模型前向传播。
4. DFL 解码。
5. 将距离转换为边界框。
6. 对分类 logits 执行 sigmoid。
7. 选择类别和对应分数。
8. 根据置信度阈值过滤候选框。
9. 执行 class-aware NMS。
10. 将坐标映射回原始图片。
11. 将坐标裁剪到图片边界内。
12. 输出检测结果。

NMS 至少需要支持：

- batch
- 多类别
- class-aware 模式
- class-agnostic 模式
- confidence threshold
- IoU threshold
- max detections
- 没有检测结果的情况
- 候选框数量过多的情况

第一版可以使用 torchvision 提供的 NMS。如果目标是完全手写，最后再替换为自己实现的 NMS。

---

## 十四、验证与指标

至少需要实现或接入：

- Precision
- Recall
- AP50
- AP50:95
- 每类别 AP
- 混淆矩阵可以后期实现

验证阶段不能使用训练标签分配逻辑，而应使用：

- 解码后的预测框
- NMS 后的检测结果
- 不同 IoU 阈值下预测框与真实框的匹配结果

为了避免自己实现 COCO AP 时产生误差，可以先接入成熟的 COCO API 验证模型，之后再手写指标。

---

## 十五、推荐实现顺序

### 阶段 1：完成模型前向传播

实现：

- Conv
- Bottleneck
- C2f
- SPPF
- Backbone
- Neck
- Detect Head 原始输出

验收：

- 三个尺度的输出形状正确
- 参数量和 FLOPs 大致符合 YOLOv8n
- 不同 batch size 均可执行前向传播

### 阶段 2：完成推理解码

实现：

- anchor points
- DFL 解码
- `dist2bbox`
- NMS
- LetterBox 坐标恢复

验收：

- 随机权重也可以完成整条图片推理流程
- 输出边界框坐标合法
- 没有检测结果时不报错

### 阶段 3：完成数据集

实现：

- YOLO 标签读取
- LetterBox
- Collate Function
- 水平翻转
- HSV 增强

验收：

- 可视化增强结果时，边界框位置仍然正确
- 空标签图片正常
- 多目标图片正常

### 阶段 4：完成标签分配和损失

实现：

- Task-Aligned Assigner
- BCE 分类损失
- CIoU loss
- DFL loss

验收：

- loss 能够 backward
- 不出现 NaN
- 空标签 batch 正常

### 阶段 5：完成最小训练闭环

实现：

- 优化器
- 学习率调度
- AMP
- checkpoint
- EMA

最关键的验收方式：

> 使用 10～50 张图片训练，模型能够快速过拟合，训练损失持续下降，并且能够在训练图片上产生正确的检测框。

在无法过拟合小数据集之前，不要加入复杂增强或多 GPU 训练。

### 阶段 6：补齐完整增强和验证

实现：

- Mosaic
- 随机仿射
- 训练后期关闭 Mosaic
- AP50
- AP50:95
- best checkpoint

### 阶段 7：与官方 YOLOv8 对齐

依次比较：

- 模型结构
- 每层输出形状
- 参数量
- FLOPs
- 权重初始化
- Detect Head bias 初始化
- anchor points
- DFL 解码结果
- 标签分配结果
- box、cls、dfl 三项损失
- NMS 结果
- 中间特征
- 训练损失曲线
- 验证指标

---

## 十六、必要的测试

### 1. 模型形状测试

检查每一层的输入输出通道和特征图尺寸。

### 2. 边界框格式转换测试

将 `xywh` 转成 `xyxy`，再转回 `xywh`，结果应基本一致。

### 3. DFL 测试

使用人工构造的分布，检查解码后的期望值是否正确。

### 4. bbox2dist 与 dist2bbox 测试

在没有触发距离裁剪的情况下，编码后再解码应能恢复原框。

### 5. IoU 测试

- 相同框的 IoU 应为 1
- 完全不相交的框的 IoU 应为 0
- 部分重叠框应与人工计算结果一致

### 6. Assigner 测试

覆盖：

- 单个目标
- 多个目标
- 目标重叠
- 匹配冲突
- 空标签
- 极小目标

### 7. Loss 测试

检查：

- loss 为有限值
- 可以执行 backward
- 梯度不为 NaN 或 Inf
- 空标签 batch 不崩溃

### 8. 数据增强测试

可视化每一种增强后的图片和边界框，确认二者同步变化。

### 9. 小数据集过拟合测试

这是验证模型、标签分配、损失和训练循环是否正确的最重要测试。

### 10. 推理坐标恢复测试

经过 LetterBox、模型预测和坐标还原后，边界框应准确映射回原图。

---

## 十七、第一版真正必要的代码

第一版只需要完成以下内容：

- Conv
- Bottleneck
- C2f
- SPPF
- Backbone
- PAN-FPN Neck
- Anchor-Free Detect Head
- anchor points 生成
- DFL 解码
- 边界框转换
- IoU 和 CIoU
- Task-Aligned Assigner
- BCE 分类损失
- CIoU 回归损失
- DFL 损失
- YOLO 标签读取
- LetterBox
- Collate Function
- 最小训练循环
- NMS
- 最小验证流程
- 小数据集过拟合测试

---

## 十八、总体原则

1. 先完成模型前向传播，再实现训练。
2. 每完成一个模块，立即编写最小测试。
3. 先保证正确，再追求训练速度和指标。
4. 先实现简单增强，再实现 Mosaic。
5. 先完成单 GPU，再考虑 DDP。
6. 先证明模型可以过拟合小数据集，再训练完整数据集。
7. 对坐标转换、DFL、Task-Aligned Assigner 和损失函数投入最多的测试精力。

手写 YOLOv8 时，最难的部分通常不是 Backbone，而是：

- 坐标系统的一致性
- DFL 编码、损失和解码
- Task-Aligned Assigner
- 正负样本与损失归一化
- 数据增强后的边界框同步变换
- 推理结果向原图坐标的恢复
