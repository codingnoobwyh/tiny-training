# RAM 优化策略
## Layerwise update

原始实现：
```
CE loss
│
fc2 backward
│
fc1 backward
│
conv2 backward
│
conv1 backward
│
统一更新权重
```
优化实现：反传完一层立马更新该层梯度，然后释放`dW`和`dB`
```
CE loss
│
│
│
fc2 backward───update fc2 
                   │
        free(dW, dB, dY)
┌────────dX────────┘
relu3 backward
│
│
│
fc1 backward───update fc1
                   │
        free(dW, dB, dY)
┌────────dX────────┘
flatten backward
│
│
│
poll2 backward
│
│
│
conv2 backward───update conv2
                   │
        free(dW, dB, dY)
┌────────dX────────┘
pool1 backward
│
│
│
relu1 backward
│
│
│
conv1 backward───update conv1
```
由于`dx`要由更新前的权重计算而来（除非可以in-place更新），所以`dx`的计算必须放在update之前，意味着`dx`和`dy`不可避免的需要共存于RAM中；`dB`和`dW`可以复用一块buffer
$$
  \begin{align}
  \text{bytes}_\text{grad\_io} &= 4 \cdot \max_{l \in L} \left( |dy_l| + |dx_l| \right) \\
  \text{bytes}_\text{grad\_wb} &= 4 \cdot \max(\max_{l \in L}(|dW_l|),\ \max_{l \in L}(|dB_l|))
  \end{align}
$$
理论上同一层的`dB`和`dW`也可以复用，但是由于`dB`的RAM占用一般很小，暂时先让`dB`和`dW`共存，避免拆分很多函数
## FC Stream Update
