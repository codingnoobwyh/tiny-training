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

FC前向:`Y = X × W + B`，其中：
```
X:  [N, K]
W:  [K, M]
B:  [M]
Y:  [N, M]
```
反传时已知`dY`，求`dX = dY × W^T`、`dW = X^T × dY`、`dB = reduce_sum(dY, axis = 0)`，如果一次性计算完整 `dW`再进行更新，会占用大量RAM资源`K × M bytes`

- 按输出通道分块
  如果`M`很大，可以每次只算一部分输出通道，此时`dB`也天然分块：
  ```
  for m0 in range(0, M, block_m):
    m1 = min(m0 + block_m, M)

    dW_block = X.T × dY[:, m0:m1]
    dB_block = sum(dY[:, m0:m1])

    update W[:, m0:m1]
    update B[m0:m1]
  ```
- 按输入分块
  ```
  for k0 in range(0, K, block_k):
    k1 = min(k0 + block_k, K)

    dW_block = X[:, k0:k1].T × dY

    update W[k0:k1, :]
  ```
- 二维分块
  ```
  for k0 in range(0, K, block_k):
    for m0 in range(0, M, block_m):
        dW_block = X[:, k0:k1].T × dY[:, m0:m1]
        update W[k0:k1, m0:m1]
  ```
由于$\text{W}_{fc}$权重是`[output, input]`存储，按输出通道进行分块，一次只计算一个输出 tile 的 `dW` 和 `db`，随即完成该 tile 的 SGD 更新，再处理下一个 tile。`dx` 跨 tile 累加。

```
for o in range(0, output_size, TILE):
    tile = min(TILE, output_size - o)
    FcCalcGradTiled(dy, x, weight, o, tile,  dx, dw_tile, db_tile)
    FcUpdateWeightTile(weight, o, tile, dw_tile)
    FcUpdateBiasTile(bias, o, tile, db_tile)
```

优点：无需分配完整 `dW[output_size × input_size]`，只需 `dw_tile[TILE × input_size]`，峰值RAM占用由 $\mathcal{O}(O \times I)$ 降为 $\mathcal{O}(\text{TILE} \times I)$。

分块大小过小后，RAM瓶颈转移到了卷积层
| tile | fc1 tile (floats) | 瓶颈层 | RAM峰值 (bytes) | 总 buffer (bytes) | 比 tile=16 省 |
|---:|---:|:---:|---:|---:|---:|
| 16 | 4816 | fc1 | 19264 | 241708 | — |
| 8  | 2408 | fc1 | 9632  | 232076 | 9.4KB |
| 6  | 1806 | fc1 | 7224  | 229668 | 11.8KB |
| 4  | 1204 | conv2 | 5232 | 227676 | 13.7KB |
| 2  | 602  | conv2 | 5232 | 227676 | 13.7KB |
|
