# NumPy Impl 下一阶段任务

当前状态：
- `npy_impl` 的前向推理已经和 `torch_impl` 对齐
- 单样本对拍结果一致
- 下一步重心不再是前向对拍脚本, 而是反向传播实现

目标：
- 在 `npy_impl` 中实现最小可验证的反向传播链路
- 先对齐单层 backward, 再串整网 backward
- 为后续 `QAS` / `SGD` / `C` 实现铺路

## 任务 3.1：实现 loss 的 backward

要做的事情：
- 在 `npy_impl/loss.py` 中新增 `cross_entropy_backward(logits, labels)`
- 输出 `dL/dlogits`
- 保持和当前 `cross_entropy_loss` 的定义一致

验收标准：
- 对同一组 `logits + labels`
- `numpy` 算出来的 `dL/dlogits`
- 和 `torch.autograd` 的结果一致或接近

## 任务 3.2：实现简单算子的 backward

实现顺序：
1. `linear_int_backward`
2. `relu_int_backward`
3. `flatten_backward`
4. `maxpool2d_2x2_int_backward`

要做的事情：
- 在 `npy_impl/ops.py` 中为上述算子补 backward
- 明确每个 backward 的输入输出：
  - 输入梯度
  - 输入缓存
  - 输出梯度结果

验收标准：
- 每个算子都能单独做梯度对拍
- 至少对比：
  - `grad_input`
  - `grad_weight`（如有）
  - `grad_bias`（如有）

## 任务 3.3：实现卷积 backward

要做的事情：
- 在 `npy_impl/ops.py` 中实现 `conv2d_3x3_int_backward`
- 当前只支持与前向一致的固定场景：
  - `3x3`
  - `padding=1`
  - `stride=1`
  - `groups=1`
  - `dilation=1`

验收标准：
- 单层卷积 backward 对拍通过
- 至少对比：
  - `grad_input`
  - `grad_weight`
  - `grad_bias`

## 任务 3.4：给模型补 forward cache 和整网 backward

要做的事情：
- 在 `npy_impl/model.py` 中保存 forward 必需缓存
- 实现整网 `backward(dL/dlogits)`
- 按当前网络结构回传：
  - `fc2`
  - `fc1`
  - `flatten`
  - `pool2`
  - `conv2`
  - `pool1`
  - `conv1`

验收标准：
- 给定单个 batch
- 能从最终 logits 的梯度一路回传到第一层
- 每层梯度 shape 正确

## 任务 3.5：新增 backward 对拍脚本

要做的事情：
- 在 `npy_impl/debug/` 下新增反向对拍入口
- 先做单层对拍, 再做整网对拍

建议脚本：
- `compare_linear_backward.py`
- `compare_conv_backward.py`
- `compare_model_backward.py`

验收标准：
- 能打印并落盘：
  - torch 梯度
  - numpy 梯度
  - max abs diff

## 当前阶段不做的事情

先不要做：
- `QAS pre_step`
- `optimizer step`
- 参数更新
- 训练循环
- 曲线绘制

原因：
- 当前阶段目标是先把 backward 语义做对
- 不是先把训练跑起来

## 推荐执行顺序

1. `loss.py`：`cross_entropy_backward`
2. `ops.py`：`linear / relu / flatten / pool backward`
3. `ops.py`：`conv backward`
4. `model.py`：整网 backward
5. `debug/`：逐层和整网 backward 对拍

## 完成标志

这一阶段完成后, 应当满足：
- `npy_impl` 不仅前向和 `torch_impl` 对齐
- backward 也能逐层对齐
- 后续才进入：
  - SGD
  - QAS
  - 训练闭环
