schedules在训练框架里通常指学习率调度和训练周期规划
包括优化器、学习率调度、训练轮次/迭代次数等配置
 首先，我们需要明确MMDetection3D配置文件的结构：
 1. 模型配置（model）：定义网络结构、检测头、损失函数等。
 2. 数据集配置（dataset）：定义数据加载、预处理、增强等。
 3. 训练配置（schedules）：定义优化器、学习率策略、训练轮次等。
 4. 运行时配置（runtime）：定义一些与训练/测试环境相关的设置，如日志、工作目录、分布式后端、模型保存、随机种子等。
# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.003  # max learning rate

`optim_wrapper`的配置字典，它定义了优化器的封装、优化器本身以及梯度裁剪的设置
OptimWrapper`类来封装优化器。`OptimWrapper`是MMEngine中提供的一个基础封装器，它简化了优化器的调用（如执行`step`和`zero_grad`），并支持梯度裁剪等功能。

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.001, betas=(0.95, 0.99)), AdamW：适合训练具有大量参数的 Transformer 架构
    clip_grad=dict(max_norm=10, norm_type=2),
 梯度裁剪  作用：防止梯度爆炸，稳定训练过程 
     max_norm=10：
    梯度裁剪的阈值
    所有梯度向量的范数将被限制 ≤10
     norm_type=2：
    使用的范数类型
    2 表示 L2 范数（欧几里得范数）
)


*参数调度器（Param Scheduler）序列
参数调度器序列 (param_scheduler)，具体用于控制训练过程中学习率的动态变化策略。它包含两个连续的调度阶段：

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
 第一阶段：线性预热 (Warmup) 阶段
 作用：在训练初期逐步增加学习率，避免初始学习率过大导致训练不稳定
    dict(第二阶段：余弦退火衰减阶段
         作用：在主体训练阶段平滑降低学习率，帮助模型收敛到更优解
        type='CosineAnnealingLR',
        begin=0,
        T_max=40,
        end=40,
        by_epoch=True,
        eta_min=1e-5)
]
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
