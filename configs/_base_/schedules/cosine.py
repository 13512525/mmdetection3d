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
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.001, betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
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
