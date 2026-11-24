# pairwise-bitdepth-dither

## 领域

游戏、图形与多媒体

## 主题

位深抖动与噪声整形

## 说明

按轨种子、按采样率系数与跨段状态。

初始状态下 `media/bitdepth_dither.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
