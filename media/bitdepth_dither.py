"""24 位转 16 位的抖动（dither）与噪声整形核心。

修复的六类缺陷（旧实现 -> 本实现）：

1. 全局随机数 -> 按轨种子
   旧实现所有轨道共用一个全局 RNG，多轨并行导出时随机序列互相消耗，
   同一工程两次导出的噪声不同、轨与轨之间还会串味。
   现在每条轨道由稳定的轨道标识经 BLAKE2b 派生独立种子，随机流按
   **绝对采样序号** 取用（PCG64.advance），与导出顺序、并行度、
   分块方式都无关。

2. 48 kHz 系数复用 44.1 kHz 素材 -> 系数按采样率设计
   ``coeff_by_rate()`` 以目标采样率的奈奎斯特频率为基准重新设计
   FIR 误差反馈系数。可听区增益压到 1 以下，抬升带顶部锚定在
   ``min(18000, 0.80 * Nyquist)``——只随奈奎斯特向下收缩、绝不向上
   扩张；回落带按抬升带频率的固定比例收窄，并且 18 kHz 以上目标恒为
   单位增益。因此高采样率素材不会把整形噪声堆到 18 kHz 以上，转
   44.1/48 kHz 或经解码器下变换时不会折叠回可听区。

3. 静音段也加抖动 -> 静音段跳过
   数字静音（样本严格为 0 / 低于门限）不注入抖动、不做量化误差反馈，
   输出逐样本为 0，数字静音保持为零。

4. 每段独立初始化整形器 -> 状态跨段延续
   ``carry_state_across_segments()`` 把上一次量化的误差历史与绝对采样
   位置带进下一段，分段导出的结果与单次导出逐样本一致，拼接处没有
   阶跃。

5. 抖动注入在限幅之后 -> 抖动在限幅前
   管线固定为 ``抖動 -> 噪声整形/量化 -> 峰值钳位``。先加抖动再限幅，
   限幅后的样本一定落在量化范围内，不会出现“裁平后再加抖动导致峰值
   越界回绕”。

6. 峰值越界回绕 -> 峰值钳位
   ``clamp_peak()`` 按目标位深的对称满量程
   ``(2**(bits-1) - 1) / 2**(bits-1)`` 钳位，整数码值落在
   ``[-(2**(bits-1)-1), 2**(bits-1)-1]``，不会符号回绕。

对外约定：所有入口对“异常/标量”输入都返回可用的结果（契约测试以
``fn(1)`` 调用），不会抛 ``NotImplementedError``。
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

__all__ = [
    # 六个修复点
    "seed_per_track",
    "coeff_by_rate",
    "skip_silence",
    "carry_state_across_segments",
    "dither_before_limit",
    "clamp_peak",
    # 处理管线
    "convert",
    "noise_shape",
    "requantize",
    "export",
    # 校验/报告
    "validate_dither",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 辅助
    "NoiseShaper",
    "PIPELINE_ORDER",
    "DEFAULT_SOURCE_BITS",
    "DEFAULT_TARGET_BITS",
    "DEFAULT_SAMPLE_RATE",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 典型工作流：24 bit 无损母带 -> 16 bit 分发
DEFAULT_SOURCE_BITS = 24
DEFAULT_TARGET_BITS = 16
DEFAULT_SAMPLE_RATE = 48000

#: 整形滤波器阶数（FIR 误差反馈，32 阶在 8 kHz~192 kHz 都足够稳定）
SHAPER_ORDER = 32

#: 有效采样率范围；超出范围时钳到边界，非法值回落到默认值
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 384000

#: 整形噪声抬升带的最高频率；高于此频率不再堆噪声，避免解码器折叠
SHAPER_TOP_HZ = 18000.0

#: 抬升带相对奈奎斯特的上限比例（低采样率下随奈奎斯特收缩）
SHAPER_TOP_RATIO = 0.80

#: 低音区噪声增益（< 1 表示整形器把量化噪声从可听区搬走）
SHAPER_AUDIO_GAIN = 0.50

#: 抬升带目标增益（> 1 表示噪声被搬到这里）
SHAPER_MID_GAIN = 1.50

#: 设计时的频域网格点数与各频段权重。抬升带之上（含 18 kHz 以上一直
#: 到奈奎斯特）保持较高权重，拟合才能把那里的增益压在单位增益附近，
#: 避免整形噪声在 18 kHz 以上继续抬升。
_DESIGN_GRID = 8192
_W_AUDIO = 20.0
_W_TRANS = 6.0
_W_GUARD = 60.0

#: 正确管线顺序：抖动必须在限幅之前
PIPELINE_ORDER = ("silence_skip", "dither", "noise_shape", "quantize", "limit", "cast")

#: 数字静音门限（以目标位深 LSB 为单位的默认值，0 表示严格为零）
SILENCE_THRESHOLD_LSB = 0.0

_EPS = 1e-15


# ---------------------------------------------------------------------------
# 基础工具：输入归一化 / 量化参数
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _coerce_bits(bits, default=DEFAULT_TARGET_BITS) -> int:
    """把任意输入收敛到合法位深（1..32）。"""
    try:
        b = int(bits)
    except (TypeError, ValueError, OverflowError):
        return default
    if b < 1:
        return 1
    if b > 32:
        return 32
    return b


def _coerce_sample_rate(sample_rate) -> int:
    """把任意输入收敛到合法的、已取整的采样率。"""
    try:
        fs = int(round(float(sample_rate)))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return DEFAULT_SAMPLE_RATE
    if fs <= 0:
        return DEFAULT_SAMPLE_RATE
    return int(min(MAX_SAMPLE_RATE, max(MIN_SAMPLE_RATE, fs)))


def _raw_array(value):
    """尽量把输入变成未清洗的原始 ndarray（用于校验 NaN/Inf）。

    与 :func:`_as_signal` 不同，这里 **不做** ``nan_to_num``；无法转换时
    返回 None，但 dict 仍会取出其中的信号字段。
    """
    if isinstance(value, dict):
        for key in ("float", "samples", "data", "signal", "audio", "output"):
            if key in value:
                return _raw_array(value[key])
        return None
    if isinstance(value, NoiseShaper):
        return np.zeros(0, dtype=np.float64)
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind in "USV" or arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
    return np.asarray(arr).ravel()


def _as_signal(value, from_bits=None) -> np.ndarray:
    """把各种输入统一成一维 float64 归一化信号（满量程 = ±1.0）。

    支持的输入：
    - 标量/列表/ndarray：浮点按已归一化处理；整数按 ``from_bits`` 位
      PCM 码值解释；
    - dict：优先取 ``samples`` / ``data`` / ``signal`` / ``float`` 字段；
    - 其他对象：尽量 ``np.asarray``，失败则返回空数组。

    归一化信号统一为 float64、一维（多声道按最后一维展开）。
    """
    if isinstance(value, dict):
        for key in ("float", "samples", "data", "signal", "audio", "output"):
            if key in value:
                return _as_signal(value[key], from_bits=from_bits)
        return np.zeros(0, dtype=np.float64)

    if isinstance(value, NoiseShaper):
        return np.zeros(0, dtype=np.float64)

    try:
        arr = np.asarray(value)
    except Exception:
        return np.zeros(0, dtype=np.float64)

    # 非数值 dtype（字符串、混合对象等）不是音频数据：视为空输入，
    # 而不是让 numpy 在 astype 时抛 ValueError
    if arr.dtype.kind in "USV" or arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError, OverflowError):
            return np.zeros(0, dtype=np.float64)

    arr = np.asarray(arr, dtype=np.float64)
    arr = np.ravel(arr)
    if arr.size == 0:
        return arr.astype(np.float64)

    # 整数 PCM：按 from_bits 的满量程归一化
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        src_bits = _coerce_bits(from_bits if from_bits is not None else DEFAULT_SOURCE_BITS)
        arr = arr / float(1 << (src_bits - 1))

    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


def _quant_step(bits: int) -> float:
    """目标位深下 1 LSB 对应的归一化幅度。"""
    return 1.0 / float(1 << (bits - 1))


def _max_code(bits: int) -> int:
    """对称满量程整数码值（正负同幅，避免正半轴越界回绕）。"""
    return (1 << (bits - 1)) - 1


def _symmetric_limit(bits: int) -> float:
    """对称满量程的归一化上限：``1 - 1 LSB``。"""
    return _max_code(bits) * _quant_step(bits)


# ---------------------------------------------------------------------------
# 1. 随机按轨种子
# ---------------------------------------------------------------------------

def _canonical_track_id(track) -> str:
    """生成稳定、可复现的轨道标识字符串。

    绝不能使用 Python 内置 ``hash()``（进程间随机化），也不能依赖
    轨道在列表中的下标，否则同一条轨道在不同导出会话里会拿到不同种子。
    """
    if track is None:
        return "track:default"
    if isinstance(track, dict):
        # 单个标识字段（如 {"id": "kick"}）与该值本身等价，避免同一轨道
        # 因调用方式不同而拿到不同种子
        identifying = [k for k in ("id", "track_id", "uid", "name", "path", "title")
                       if k in track]
        if len(identifying) == 1 and len(track) == 1:
            return _canonical_track_id(track[identifying[0]])
        for key in identifying:
            return f"track:{key}={_canonical_track_id(track[key])}"
        items = sorted((str(k), _canonical_track_id(v)) for k, v in track.items())
        return "track:{" + ",".join(f"{k}={v}" for k, v in items) + "}"
    if isinstance(track, (list, tuple)):
        return "track:[" + ",".join(_canonical_track_id(v) for v in track) + "]"
    if isinstance(track, bytes):
        return "track:bytes:" + track.hex()
    if isinstance(track, (np.integer, np.floating)):
        track = track.item()
    if isinstance(track, float):
        # 避免 -0.0 / 0.0 以及 NaN 造成的不稳定字符串
        if math.isnan(track):
            return "track:float:nan"
        return f"track:float:{track!r}"
    return f"track:{track!r}"


def seed_per_track(track=None, salt: str = "bitdepth-dither") -> int:
    """由轨道标识派生稳定的 64 位随机种子。

    同一轨道在任意机器、任意进程、任意导出顺序下都得到相同种子；
    不同轨道（以及不同 ``salt``）得到互不相关的种子。

    :param track: 轨道标识（int/str/dict/路径等均可）。
    :param salt:  域分隔字符串，可用于区分“抖动/整形/噪声底”等不同用途。
    :return: 0 <= seed < 2**64 的整数。
    """
    domain = f"{salt}|v1|".encode("utf-8")
    payload = _canonical_track_id(track).encode("utf-8")
    digest = hashlib.blake2b(domain + payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# 2. 系数按采样率
# ---------------------------------------------------------------------------

def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _db_mix(g0: float, g1: float, t):
    """在 dB 域做平滑插值，避免线性插值造成的响应折点。"""
    db0 = 20.0 * math.log10(max(g0, _EPS))
    db1 = 20.0 * math.log10(max(g1, _EPS))
    return 10.0 ** ((db0 + (db1 - db0) * _smoothstep(t)) / 20.0)


def _shaper_bands(sample_rate: int):
    """返回 (f_audio, f_mid, f_hold, f_guard)，单位 Hz。

    - ``f_audio`` 以下：噪声增益 < 1，量化噪声被搬离可听区；
    - ``f_audio ~ f_mid``：平滑抬升到 ``SHAPER_MID_GAIN``；
    - ``f_mid ~ f_hold``：保持抬升（否则过渡带太窄，拟合会欠阻尼）；
    - ``f_hold ~ f_guard``：回落到单位增益；
    - ``f_guard`` 以上：保持 1.0，不再堆噪声。

    两个关键约束：
    1. 抬升带顶部固定以 18 kHz 为锚点（``SHAPER_TOP_HZ``），只随奈奎斯特
       频率向下收缩，绝不向上扩张；
    2. 回落带的宽度按抬升带频率的固定比例收窄（``f_mid`` 的 0.15 倍），
       而不是铺满剩余带宽。

    如果按奈奎斯特比例铺开，96 kHz 素材的整形噪声会一直抬到 37 kHz，
    转成 48 kHz 或经解码器下变换时全部折叠回可听区——这正是要修的问题。
    """
    nyquist = sample_rate / 2.0
    f_audio = min(13500.0, 0.55 * nyquist)
    f_mid = min(SHAPER_TOP_HZ, SHAPER_TOP_RATIO * nyquist)
    f_mid = max(f_mid, f_audio * 1.05)
    f_hold = min(0.96 * nyquist, f_mid * 1.08)
    f_guard = min(0.98 * nyquist, f_hold + f_mid * 0.15)
    f_guard = max(f_guard, f_hold * 1.02)
    return f_audio, f_mid, f_hold, f_guard


@lru_cache(maxsize=64)
def _design_coeffs(sample_rate: int, order: int) -> tuple:
    """按目标采样率设计 FIR 误差反馈系数（NTF = 1 + C(z)）。

    目标幅频响应：
    - 可听区增益 ``SHAPER_AUDIO_GAIN``（把量化噪声搬走）；
    - 抬升带增益 ``SHAPER_MID_GAIN``，带顶 = ``min(18000, 0.80 * Nyquist)``；
    - 奈奎斯特附近回到 1.0，避免整形噪声堆在 18 kHz 以上。

    用加权复数最小二乘拟合 FIR 频率响应；系数只与采样率有关，并缓存。
    """
    nyquist = sample_rate / 2.0
    f_audio, f_mid, f_hold, f_guard = _shaper_bands(sample_rate)

    freqs = np.linspace(0.0, nyquist, _DESIGN_GRID, endpoint=False)
    target = np.full_like(freqs, SHAPER_AUDIO_GAIN)

    band = (freqs >= f_audio) & (freqs < f_mid)
    if np.any(band):
        t = (freqs[band] - f_audio) / max(f_mid - f_audio, _EPS)
        target[band] = _db_mix(SHAPER_AUDIO_GAIN, SHAPER_MID_GAIN, t)

    target[(freqs >= f_mid) & (freqs < f_hold)] = SHAPER_MID_GAIN

    guard = freqs >= f_hold
    if np.any(guard):
        t = np.clip((freqs[guard] - f_hold) / max(f_guard - f_hold, _EPS), 0.0, 1.0)
        target[guard] = _db_mix(SHAPER_MID_GAIN, 1.0, t)

    # 硬锚点：SHAPER_TOP_HZ 以上目标恒为单位增益。回落带若只是“接近”
    # 1.0，拟合误差会让 18 kHz 以上仍然抬升几个 dB，下变换时折叠回可听区。
    anchor = freqs >= SHAPER_TOP_HZ
    if np.any(anchor):
        target[anchor] = 1.0

    weights = np.full_like(freqs, _W_AUDIO)
    weights[band] = _W_TRANS
    weights[freqs >= f_mid] = _W_GUARD

    omega = 2.0 * np.pi * freqs / float(sample_rate)
    basis = np.exp(-1j * np.outer(omega, np.arange(1, order + 1)))
    rhs = (target - 1.0) * weights
    basis *= weights[:, None]

    # 复数最小二乘 -> 堆叠实部/虚部求实数系数
    system = np.vstack([basis.real, basis.imag])
    target_vec = np.concatenate([rhs.real, rhs.imag])
    coeffs, *_ = np.linalg.lstsq(system, target_vec, rcond=None)

    # 数值清洗：去除浮点尘埃，保证缓存/比较稳定
    coeffs = np.round(coeffs, 12)
    return tuple(float(c) for c in coeffs)


def coeff_by_rate(sample_rate=DEFAULT_SAMPLE_RATE, order=SHAPER_ORDER) -> np.ndarray:
    """按采样率返回噪声整形 FIR 系数（``NTF = 1 + C(z)``）。

    44.1 kHz 与 48 kHz 素材不能共用一套按 48 kHz 设计的系数：系数按
    目标采样率的奈奎斯特频率重新设计，抬升带落在
    ``min(18 kHz, 0.8 * Nyquist)`` 以内，高采样率下也不会把噪声堆到
    18 kHz 以上。

    :param sample_rate: 目标采样率（非法值回落到默认值，范围外钳制）。
    :param order: 滤波器阶数（1..128）。
    :return: 长度为 ``order`` 的 float64 ndarray。
    """
    fs = _coerce_sample_rate(sample_rate)
    try:
        k = int(order)
    except (TypeError, ValueError, OverflowError):
        k = SHAPER_ORDER
    k = int(min(128, max(1, k)))
    return np.asarray(_design_coeffs(fs, k), dtype=np.float64)


def _shaper_response(coeffs, sample_rate, freqs=None) -> tuple:
    """返回 (freqs, |NTF|)，用于校验与审计。

    ``NTF = 1 + C(z)``，与处理管线中的误差反馈方向一致。
    """
    fs = _coerce_sample_rate(sample_rate)
    coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
    if freqs is None:
        freqs = np.linspace(0.0, fs / 2.0, 4097)
    else:
        freqs = np.asarray(freqs, dtype=np.float64).ravel()
    if coeffs.size == 0:
        return freqs, np.ones_like(freqs)
    omega = 2.0 * np.pi * freqs / float(fs)
    basis = np.exp(-1j * np.outer(omega, np.arange(1, coeffs.size + 1)))
    ntf = 1.0 + basis @ coeffs
    return freqs, np.abs(ntf)


# ---------------------------------------------------------------------------
# 3. 静音段跳过
# ---------------------------------------------------------------------------

def skip_silence(samples, threshold_lsb=SILENCE_THRESHOLD_LSB,
                 bits=DEFAULT_TARGET_BITS, from_bits=None) -> np.ndarray:
    """返回“数字静音、应跳过抖动”的布尔掩码。

    数字静音定义为样本严格为 0，或幅度不超过
    ``threshold_lsb * 1 LSB``（默认门限 0，即只认严格零样本）。
    被标记的样本既不注入抖动也不参与量化误差反馈，输出保持逐样本 0，
    数字静音因此不会变成非零噪声。

    :return: 与输入等长的 bool ndarray（标量输入返回长度 1 的数组）。
    """
    x = _as_signal(samples, from_bits=from_bits)
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    target_bits = _coerce_bits(bits)
    try:
        threshold = float(threshold_lsb)
    except (TypeError, ValueError, OverflowError):
        threshold = 0.0
    threshold = max(0.0, threshold)

    limit = threshold * _quant_step(target_bits)
    return np.abs(x) <= limit


def _silence_mask(x: np.ndarray, mask=None) -> np.ndarray:
    """统一静音掩码：未提供则按严格零判定。"""
    if mask is None:
        return x == 0.0
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != x.size:
        return x == 0.0
    return mask


# ---------------------------------------------------------------------------
# 按绝对采样序号生成抖动（分段不变）
# ---------------------------------------------------------------------------

def _dither_block(seed: int, start: int, count: int,
                  lsb: float, silent: np.ndarray) -> np.ndarray:
    """生成 ``[start, start+count)`` 区间的 TPDF 抖动，幅度 ±1 LSB。

    随机数按 **绝对采样序号** 取用：每个样本恰好消耗两个 float64，
    PCG64 每产生一个 float64 消耗一个 64 位输出，因此
    ``advance(start * 2)`` 之后连续取两个均匀数 ``u1, u2`` 就是该样本的
    TPDF 三角分布 ``(u1 - u2) * 1 LSB``。

    注意必须按“每样本连续两个”切片，而不是把整块分成前后两半相减：
    后者在分块导出时取到的是不同的随机数对，噪声会随分块方式变化。
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)

    bit_generator = np.random.PCG64(int(seed) & ((1 << 64) - 1))
    bit_generator.advance(int(max(0, start)) * 2)
    generator = np.random.Generator(bit_generator)
    draws = generator.random(2 * count).reshape(count, 2)
    triangular = (draws[:, 0] - draws[:, 1]) * lsb
    triangular[silent] = 0.0
    return triangular


# ---------------------------------------------------------------------------
# 整形器状态（跨段延续）
# ---------------------------------------------------------------------------

def _fresh_state(order: int, position: int = 0) -> dict:
    return {
        "order": int(order),
        "error": [0.0] * int(order),  # 最近一次在 index 0
        "position": int(position),
    }


def _state_from(state, order: int, position: int = 0) -> dict:
    """把用户传入的状态（dict/None）转换为内部状态。"""
    if isinstance(state, dict):
        errors = state.get("error", state.get("errors", None))
        if errors is None:
            errors = [0.0] * order
    else:
        errors = None

    if errors is None:
        return _fresh_state(order, position)

    errors = [float(v) for v in list(errors)[:order]]
    if len(errors) < order:
        errors = errors + [0.0] * (order - len(errors))
    pos = state.get("position", position)
    try:
        pos = int(pos)
    except (TypeError, ValueError, OverflowError):
        pos = int(position)
    return {"order": int(order), "error": errors, "position": max(0, pos)}


def _state_public(state: dict) -> dict:
    """返回可序列化的状态副本（list，而不是 ndarray）。"""
    return {
        "order": int(state.get("order", 0)),
        "error": [float(v) for v in state.get("error", [])],
        "position": int(state.get("position", 0)),
    }


class NoiseShaper:
    """有状态的误差反馈噪声整形器。

    处理公式（误差反馈，``NTF = 1 + C(z)``）::

        y[n] = x[n] + d[n] + sum_k c[k] * e[n-k]
        q[n] = quantize(y[n])
        e[n] = q[n] - y[n]

    状态里保存 ``e[n-k]`` 与 **绝对采样位置**；分段处理时调用
    :meth:`process` 并传入上一次的 ``state`` 即可无缝续接。
    """

    def __init__(self, coeffs=None, sample_rate=DEFAULT_SAMPLE_RATE,
                 bits=DEFAULT_TARGET_BITS, state=None):
        self.sample_rate = _coerce_sample_rate(sample_rate)
        self.bits = _coerce_bits(bits)
        self.coeffs = (np.asarray(coeffs, dtype=np.float64).ravel()
                       if coeffs is not None else coeff_by_rate(self.sample_rate))
        self.state = _state_from(state, self.coeffs.size, 0)

    def process(self, samples, dither=None, mask=None):
        """处理一段信号，返回 ``(float_out, state)``（不改变对象状态）。"""
        x = _as_signal(samples)
        silent = _silence_mask(x, mask)
        if dither is None:
            dither = np.zeros_like(x)
        else:
            dither = np.asarray(dither, dtype=np.float64).ravel()
            if dither.size != x.size:
                dither = np.resize(dither, x.size)

        out, state, _ = _shape_quantize(
            x, dither, silent, self.coeffs, self.state, self.bits)
        return out, state

    @property
    def state_dict(self) -> dict:
        return _state_public(self.state)


def _shape_quantize(x: np.ndarray, dither: np.ndarray, silent: np.ndarray,
                    coeffs: np.ndarray, state: dict, bits: int):
    """误差反馈 + 量化，返回 ``(输出 float, 新状态, 钳位样本数)``。

    - 静音样本：输出 0、误差 0，并照常推进历史与位置；
    - 峰值在量化码值层面钳位，误差用钳位后的码值计算，避免限幅引起
      的极限环；
    - 全部为标量运算，保证逐样本可复现。

    逐样本循环的写法（``math.fsum(map(mul, ...))`` + 内联四舍五入）是
    有意的：内核必须严格顺序执行才能保证跨段、跨平台逐位一致，所以不能
    换成向量化的块处理。
    """
    n = x.size
    inv_q = float(1 << (bits - 1))          # 1 / q
    q = _quant_step(bits)
    max_code = _max_code(bits)
    order = int(coeffs.size)
    errors = [float(v) for v in state.get("error", ())][:order]
    errors += [0.0] * (order - len(errors))

    coeff_list = [float(c) for c in coeffs]
    fsum = math.fsum
    mul = operator.mul
    out = np.empty(n, dtype=np.float64)
    out_list = [0.0] * n
    position = int(state.get("position", 0))
    clipped = 0

    xv = x.tolist()
    dv = dither.tolist()
    sv = silent.tolist()

    for i in range(n):
        if sv[i]:
            # 数字静音：不抖动、不反馈，保持精确零
            err = 0.0
            value = 0.0
        else:
            feedback = fsum(map(mul, coeff_list, errors))
            y = xv[i] + dv[i] + feedback
            scaled = y * inv_q
            # 四舍五入（远离零），避免银行家舍入带来的偏置
            if scaled >= 0.0:
                code = int(scaled + 0.5)
                if code > max_code:
                    code = max_code
                    clipped += 1
            else:
                code = -int(0.5 - scaled)
                if code < -max_code:
                    code = -max_code
                    clipped += 1
            value = code * q
            err = value - y
        out_list[i] = value

        # 推进误差历史（index 0 = 最近）
        errors.pop()
        errors.insert(0, err)

    out[:] = out_list
    new_state = {
        "order": order,
        "error": errors,
        "position": position + n,
    }
    return out, new_state, clipped


# ---------------------------------------------------------------------------
# 4. 状态跨段延续
# ---------------------------------------------------------------------------

def _split_segments(length: int, segments):
    """把 ``segments`` 归一化成 [(start, end), ...] 覆盖 [0, length)。"""
    if length <= 0:
        return []

    if segments is None:
        return [(0, length)]

    if isinstance(segments, (int, np.integer)) and not isinstance(segments, bool):
        k = int(segments)
        if k <= 1:
            return [(0, length)]
        k = min(k, length)
        edges = np.linspace(0, length, k + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]

    try:
        pairs = [(int(a), int(b)) for a, b in segments]
    except (TypeError, ValueError):
        return [(0, length)]

    pairs = [(max(0, a), min(length, b)) for a, b in pairs if b > a]
    if not pairs:
        return [(0, length)]

    # 显式分段只覆盖了一部分时，把缺口补成分段，而不是把未处理的样本
    # 留成零（那会在工程尾部凭空造出一段静音）
    pairs.sort()
    covered = []
    cursor = 0
    for start, end in pairs:
        if start > cursor:
            covered.append((cursor, start))
        covered.append((start, max(start, end)))
        cursor = max(cursor, end)
    if cursor < length:
        covered.append((cursor, length))
    return [(a, b) for a, b in covered if b > a]


def carry_state_across_segments(samples, segments=None, *, track=None,
                                seed=None, sample_rate=DEFAULT_SAMPLE_RATE,
                                bits=DEFAULT_TARGET_BITS, coeffs=None,
                                state=None, dither=True,
                                threshold_lsb=SILENCE_THRESHOLD_LSB) -> dict:
    """分段处理并让整形器状态跨段延续，返回拼接结果与诊断信息。

    旧实现每个分段都 ``state = zeros(order)``，段边界处噪声整形突然
    重启，拼接处出现阶跃。这里每段处理完都把误差历史与绝对采样位置
    传给下一段，因此：

    - 分段导出的拼接结果与单次整段导出 **逐样本一致**；
    - 段边界处的样本差值与整段导出相同（没有人为阶跃）。

    输入可以是：
    - 一整段信号 + ``segments``（列表 [(start, end)] 或分块数量 int）；
    - 分段信号的列表（每段为独立数组，按顺序拼接）。

    返回 dict：
    ``output``（float 归一化）、``state``、``segment_states``、
    ``segments``、``seam_max_delta``、``single_pass_equal``、``seed``。

    对无法解析的输入（如 ``carry_state_across_segments(1)``）退化为
    单样本/单段处理，始终返回可用的报告。
    """
    target_bits = _coerce_bits(bits)
    fs = _coerce_sample_rate(sample_rate)
    coeff_arr = (np.asarray(coeffs, dtype=np.float64).ravel()
                 if coeffs is not None else coeff_by_rate(fs))

    track_seed = int(seed) if seed is not None else seed_per_track(
        track if track is not None else samples, salt="dither")

    # 情况 A：输入本身是分段列表
    chunks = None
    if isinstance(samples, (list, tuple)) and samples and all(
            isinstance(item, (list, tuple, np.ndarray)) for item in samples):
        chunks = [_as_signal(item) for item in samples]
        x = np.concatenate(chunks) if chunks else np.zeros(0)
        ranges = []
        cursor = 0
        for chunk in chunks:
            ranges.append((cursor, cursor + chunk.size))
            cursor += chunk.size
    else:
        x = _as_signal(samples)
        ranges = _split_segments(x.size, segments)

    silent_all = skip_silence(x, threshold_lsb=threshold_lsb,
                              bits=target_bits) if x.size else np.zeros(0, dtype=bool)

    current = _state_from(state, coeff_arr.size, 0)
    outputs = []
    segment_states = []
    for start, end in ranges:
        block = x[start:end]
        silent = silent_all[start:end]
        if dither:
            noise = _dither_block(track_seed, start, block.size,
                                  _quant_step(target_bits), silent)
        else:
            noise = np.zeros(block.size, dtype=np.float64)
        rendered, current, _ = _shape_quantize(
            block, noise, silent, coeff_arr, current, target_bits)
        outputs.append(rendered)
        segment_states.append({
            "start": int(start),
            "end": int(end),
            "state": _state_public(current),
        })

    output = np.concatenate(outputs) if outputs else np.zeros(0)
    if output.size != x.size:
        # 段未覆盖全部样本时补零，保证输出长度与输入一致
        padded = np.zeros(x.size, dtype=np.float64)
        padded[:min(output.size, x.size)] = output[:min(output.size, x.size)]
        output = padded

    # 单次整段处理的对照结果（用于验证分段不变性）
    if dither:
        dither_all = _dither_block(track_seed, 0, x.size,
                                   _quant_step(target_bits), silent_all)
    else:
        dither_all = np.zeros(x.size, dtype=np.float64)
    single, _, _ = _shape_quantize(
        x, dither_all, silent_all, coeff_arr, _state_from(None, coeff_arr.size, 0),
        target_bits)

    single_pass_equal = bool(np.array_equal(output, single)) if x.size else True

    seam_max_delta = 0.0
    for _, boundary in ranges[:-1]:
        if 0 < boundary < output.size:
            seam_max_delta = max(
                seam_max_delta, float(abs(output[boundary] - output[boundary - 1])))

    return {
        "output": output,
        "state": _state_public(current),
        "segment_states": segment_states,
        "segments": [[int(a), int(b)] for a, b in ranges],
        "seam_max_delta": seam_max_delta,
        "single_pass_equal": single_pass_equal,
        "seed": track_seed,
        "order": int(coeff_arr.size),
        "sample_rate": fs,
        "bits": target_bits,
    }


# ---------------------------------------------------------------------------
# 5/6. 抖动在限幅前 + 峰值钳位
# ---------------------------------------------------------------------------

def _quantize(x: np.ndarray, bits: int):
    """把连续值量化到目标位深的码值格点，返回 (float, 超量程码值数)。

    这一步在“不加整形/不加抖动”的路径上同样必须执行，否则重量化等于
    直通，``float`` 与 ``codes`` 会不一致。
    """
    if x.size == 0:
        return x.astype(np.float64), 0
    inv_q = float(1 << (bits - 1))
    max_code = _max_code(bits)
    codes = np.rint(x * inv_q)
    over = int(np.count_nonzero(np.abs(codes) > max_code))
    np.clip(codes, -max_code, max_code, out=codes)
    return codes / inv_q, over


def _clamp_with_count(x: np.ndarray, bits: int, ceiling=None):
    """对称满量程钳位，返回 (钳位结果, 越界样本数)。"""
    limit = _symmetric_limit(bits)
    if ceiling is not None:
        try:
            limit = min(limit, abs(float(ceiling)))
        except (TypeError, ValueError, OverflowError):
            pass
    if x.size == 0:
        return x.astype(np.float64), 0
    clipped = int(np.count_nonzero(np.abs(x) > limit))
    return np.clip(x, -limit, limit), clipped


def clamp_peak(samples, ceiling=None, bits=DEFAULT_TARGET_BITS,
               from_bits=None) -> np.ndarray:
    """按目标位深的对称满量程钳位，杜绝峰值回绕。

    上限为 ``(2**(bits-1) - 1) / 2**(bits-1)``（即 1 - 1 LSB）：正负
    半轴同幅，转成整数后一定落在
    ``[-(2**(bits-1)-1), 2**(bits-1)-1]``，不会因为 +32768 越界而
    回绕成负值。``ceiling`` 可再收紧一个线性上限（如 0.999）。

    :return: 与输入同形状的 float64 ndarray（归一化）。
    """
    x = _as_signal(samples, from_bits=from_bits)
    out, _ = _clamp_with_count(x, _coerce_bits(bits), ceiling)
    return out


def dither_before_limit(samples, *, bits=DEFAULT_TARGET_BITS,
                        from_bits=None, track=None, seed=None,
                        sample_rate=DEFAULT_SAMPLE_RATE, ceiling=None,
                        shape=True, threshold_lsb=SILENCE_THRESHOLD_LSB,
                        state=None) -> np.ndarray:
    """按正确顺序处理一段信号：抖动 -> 整形/量化 -> 限幅。

    关键点是 **抖动在限幅之前**：旧实现先限幅（把尾部样本裁平）再加
    抖动，抖动会把已裁平的峰值重新推出量化范围，整数转换时回绕。
    现在先加抖动、再量化、最后钳位，输出的幅度上界由钳位保证。

    数字静音样本按 :func:`skip_silence` 跳过，保持精确零。

    :return: 限幅后的 float64 归一化 ndarray。
    """
    target_bits = _coerce_bits(bits)
    fs = _coerce_sample_rate(sample_rate)
    x = _as_signal(samples, from_bits=from_bits)
    if x.size == 0:
        return x

    track_seed = int(seed) if seed is not None else seed_per_track(
        track if track is not None else samples, salt="dither")
    silent = skip_silence(x, threshold_lsb=threshold_lsb, bits=target_bits)

    if shape:
        coeff_arr = coeff_by_rate(fs)
        start_state = _state_from(state, coeff_arr.size, 0)
        # 抖动偏移跟随状态的绝对采样位置，续接时噪声不断档
        noise = _dither_block(track_seed, start_state["position"], x.size,
                              _quant_step(target_bits), silent)
        rendered, _, _ = _shape_quantize(
            x, noise, silent, coeff_arr, start_state, target_bits)
    else:
        noise = _dither_block(track_seed, 0, x.size,
                              _quant_step(target_bits), silent)
        # 不整形也要量化：抖动后必须落回目标位深的码值格点
        rendered, _ = _quantize(np.where(silent, 0.0, x + noise), target_bits)

    out, _ = _clamp_with_count(rendered, target_bits, ceiling)
    return out


# ---------------------------------------------------------------------------
# 核心管线：噪声整形 / 重量化 / 转换
# ---------------------------------------------------------------------------

def _render(x: np.ndarray, *, bits: int, sample_rate: int, seed: int,
            coeffs=None, shape: bool = True, dither: bool = True,
            threshold_lsb: float = SILENCE_THRESHOLD_LSB,
            state=None, start: int = 0):
    """内部渲染：抖动 -> (整形) -> 量化 -> 钳位。返回 (float, state, 诊断)。"""
    silent = skip_silence(x, threshold_lsb=threshold_lsb, bits=bits)
    if dither:
        noise = _dither_block(seed, start, x.size, _quant_step(bits), silent)
    else:
        noise = np.zeros(x.size, dtype=np.float64)

    if shape:
        coeff_arr = (np.asarray(coeffs, dtype=np.float64).ravel()
                     if coeffs is not None else coeff_by_rate(sample_rate))
        rendered, new_state, code_clipped = _shape_quantize(
            x, noise, silent, coeff_arr, _state_from(state, coeff_arr.size, start),
            bits)
    else:
        coeff_arr = np.zeros(0, dtype=np.float64)
        # 不做噪声整形也必须量化：否则返回的 float 仍然是未量化的连续值，
        # 与 codes 对不上，重量化实际没有发生
        unshaped = np.where(silent, 0.0, x + noise)
        rendered, code_clipped = _quantize(unshaped, bits)
        new_state = _fresh_state(0, start + x.size)

    clamped, clipped = _clamp_with_count(rendered, bits)
    codes = np.rint(clamped / _quant_step(bits)).astype(np.int64)
    return clamped, codes, new_state, {
        "skipped_silence": int(np.count_nonzero(silent)),
        "clipped": int(clipped) + int(code_clipped),
        "seed": int(seed),
        "bits": int(bits),
        "sample_rate": int(sample_rate),
        "coeffs": coeff_arr,
    }


def noise_shape(samples, coeffs=None, sample_rate=DEFAULT_SAMPLE_RATE,
                bits=DEFAULT_TARGET_BITS, state=None, from_bits=None,
                dither=False, track=None, seed=None) -> dict:
    """对信号做误差反馈噪声整形（可选同时加抖动）。

    返回 dict：
    - ``shaped``：整形并量化后的 float 归一化波形；
    - ``codes``：目标位深整数码值；
    - ``state``：可跨段延续的误差历史（含绝对位置）；
    - ``coeffs`` / ``sample_rate`` / ``bits``：实际使用的参数；
    - ``peak_hz``：该采样率下整形噪声抬升带的顶部频率。

    对 ``noise_shape(1)`` 这类单样本输入同样返回完整报告。
    """
    target_bits = _coerce_bits(bits)
    fs = _coerce_sample_rate(sample_rate)
    x = _as_signal(samples, from_bits=from_bits)
    coeff_arr = (np.asarray(coeffs, dtype=np.float64).ravel()
                 if coeffs is not None else coeff_by_rate(fs))

    track_seed = int(seed) if seed is not None else seed_per_track(
        track if track is not None else samples, salt="dither")

    if x.size == 0:
        return {
            "shaped": np.zeros(0), "codes": np.zeros(0, dtype=np.int64),
            "state": _state_public(_state_from(state, coeff_arr.size, 0)),
            "coeffs": coeff_arr, "sample_rate": fs, "bits": target_bits,
            "peak_hz": _shaper_bands(fs)[1], "seed": track_seed,
            "skipped_silence": 0,
        }

    start_state = _state_from(state, coeff_arr.size, 0)
    offset = start_state["position"]
    silent = skip_silence(x, bits=target_bits)
    if dither:
        noise = _dither_block(track_seed, offset, x.size,
                              _quant_step(target_bits), silent)
    else:
        noise = np.zeros(x.size, dtype=np.float64)

    shaped, new_state, code_clipped = _shape_quantize(
        x, noise, silent, coeff_arr, start_state, target_bits)
    clamped, clipped = _clamp_with_count(shaped, target_bits)
    codes = np.rint(clamped / _quant_step(target_bits)).astype(np.int64)
    return {
        "shaped": clamped,
        "codes": codes,
        "state": _state_public(new_state),
        "coeffs": coeff_arr,
        "sample_rate": fs,
        "bits": target_bits,
        "peak_hz": _shaper_bands(fs)[1],
        "seed": track_seed,
        "skipped_silence": int(np.count_nonzero(silent)),
        "clipped": int(clipped) + int(code_clipped),
    }


def requantize(samples, from_bits=DEFAULT_SOURCE_BITS,
               to_bits=DEFAULT_TARGET_BITS, sample_rate=DEFAULT_SAMPLE_RATE,
               track=None, seed=None, segments=None, shape=True, dither=True,
               threshold_lsb=SILENCE_THRESHOLD_LSB, state=None,
               integer=True) -> dict:
    """把 ``from_bits`` 位 PCM 重量化到 ``to_bits`` 位（默认 24 -> 16）。

    同一管线内完成：静音跳过 -> 按轨抖动 -> 按采样率整形 -> 峰值钳位。
    支持 ``segments`` 分段处理（状态自动跨段延续），也支持传入上一次的
    ``state`` 续接。

    返回 dict：
    - ``samples``：目标位深整数码值（dtype 至少 int16，必要时 int32）；
    - ``float``：归一化浮点输出（= 码值 × 1 LSB）；
    - ``state``：最终整形器状态；
    - ``skipped_silence`` / ``clipped`` / ``seed`` / ``sample_rate`` /
      ``bits`` / ``coeffs`` / ``segments`` / ``block_invariant``。
    """
    src_bits = _coerce_bits(from_bits, DEFAULT_SOURCE_BITS)
    target_bits = _coerce_bits(to_bits)
    fs = _coerce_sample_rate(sample_rate)
    x = _as_signal(samples, from_bits=src_bits)

    track_seed = int(seed) if seed is not None else seed_per_track(
        track if track is not None else samples, salt="dither")

    coeff_arr = coeff_by_rate(fs) if shape else np.zeros(0)

    if x.size == 0:
        empty_codes = np.zeros(0, dtype=np.int16 if target_bits <= 16 else np.int32)
        return {
            "samples": empty_codes, "float": np.zeros(0),
            "state": _state_public(_state_from(state, coeff_arr.size, 0)),
            "skipped_silence": 0, "clipped": 0, "seed": track_seed,
            "sample_rate": fs, "bits": target_bits, "coeffs": coeff_arr,
            "segments": [], "block_invariant": True,
        }

    ranges = _split_segments(x.size, segments)
    current = _state_from(state, coeff_arr.size, 0)
    floats = []
    diagnostics = []
    for start, end in ranges:
        block = x[start:end]
        # 抖动偏移取状态的绝对采样位置：显式续接 state 与 segments=
        # 分段两条路径必须取到同一段随机序列，否则拼接处噪声不连续
        rendered, codes, current, diag = _render(
            block, bits=target_bits, sample_rate=fs, seed=track_seed,
            coeffs=coeff_arr, shape=shape, dither=dither,
            threshold_lsb=threshold_lsb, state=current,
            start=current["position"])
        floats.append(rendered)
        diagnostics.append({
            "start": int(start), "end": int(end),
            "skipped_silence": diag["skipped_silence"],
            "clipped": diag["clipped"],
        })

    out_float = np.concatenate(floats) if floats else np.zeros(0)
    codes_all = np.rint(out_float / _quant_step(target_bits)).astype(np.int64)
    # 对称量程：正负都不越界，不存在回绕
    max_code = _max_code(target_bits)
    np.clip(codes_all, -max_code, max_code, out=codes_all)

    dtype = np.int16 if target_bits <= 16 else np.int32
    out_codes = codes_all.astype(dtype)
    skipped = int(sum(d["skipped_silence"] for d in diagnostics))
    clipped = int(sum(d["clipped"] for d in diagnostics))

    # 分段不变性：单次整段渲染应完全一致。只有真的分成多段时才需要
    # 额外的整段对照渲染（单段时结果当然相同）。
    if len(ranges) > 1:
        single_float, _, _, _ = _render(
            x, bits=target_bits, sample_rate=fs, seed=track_seed, coeffs=coeff_arr,
            shape=shape, dither=dither, threshold_lsb=threshold_lsb, start=0)
        block_invariant = bool(np.array_equal(out_float, single_float))
    else:
        block_invariant = True

    final_state = current
    return {
        "samples": out_codes if integer else out_float,
        "float": out_float,
        "state": _state_public(final_state),
        "skipped_silence": skipped,
        "clipped": clipped,
        "seed": track_seed,
        "sample_rate": fs,
        "bits": target_bits,
        "source_bits": src_bits,
        "coeffs": coeff_arr,
        "segments": [[int(a), int(b)] for a, b in ranges],
        "segment_diagnostics": diagnostics,
        "block_invariant": block_invariant,
    }


def convert(samples, from_bits=DEFAULT_SOURCE_BITS,
            to_bits=DEFAULT_TARGET_BITS, sample_rate=DEFAULT_SAMPLE_RATE,
            track=None, seed=None, segments=None, shape=True, dither=True,
            threshold_lsb=SILENCE_THRESHOLD_LSB, integer=True) -> dict:
    """24 -> 16 位主入口（:func:`requantize` 的语义别名 + 输入兼容）。

    对 ``convert(1)`` 这类标量输入按 24 bit 整数 PCM 解释，返回完整
    报告；对 dict 输入会先取出其中的样本。
    """
    return requantize(
        samples, from_bits=from_bits, to_bits=to_bits,
        sample_rate=sample_rate, track=track, seed=seed, segments=segments,
        shape=shape, dither=dither, threshold_lsb=threshold_lsb,
        integer=integer)


# ---------------------------------------------------------------------------
# 多轨导出
# ---------------------------------------------------------------------------

def _track_key(index: int, track, fallback: str = "track") -> str:
    """导出结果使用的稳定键名（用于报告，不参与种子派生）。"""
    if isinstance(track, dict):
        for key in ("id", "track_id", "uid", "name", "path", "title"):
            if key in track:
                return str(track[key])
    if isinstance(track, (str, bytes)):
        return track.decode("utf-8", "replace") if isinstance(track, bytes) else track
    return f"{fallback}[{index}]"


def _track_identity(index: int, track):
    """参与种子派生的轨道身份：优先使用显式标识，否则用下标+内容摘要。"""
    if isinstance(track, dict):
        for key in ("id", "track_id", "uid", "name", "path", "title"):
            if key in track:
                return track[key]
        signal = _as_signal(track)
        digest = hashlib.blake2b(signal.tobytes(), digest_size=8).hexdigest()
        return f"content:{digest}"
    if isinstance(track, (str, bytes, int, float, np.integer, np.floating)):
        return track
    signal = _as_signal(track)
    digest = hashlib.blake2b(signal.tobytes(), digest_size=8).hexdigest()
    return f"content:{digest}"


def export(tracks, from_bits=DEFAULT_SOURCE_BITS, to_bits=DEFAULT_TARGET_BITS,
           sample_rate=DEFAULT_SAMPLE_RATE, parallel=True, segments=None,
           shape=True, dither=True,
           threshold_lsb=SILENCE_THRESHOLD_LSB) -> dict:
    """多轨导出：每轨派生独立种子，结果与顺序/并行度无关。

    旧实现多轨共用全局 RNG，并行导出时各线程互相消耗随机序列，同一
    工程两次导出的噪声不同。现在：

    - 每条轨道由稳定的轨道身份派生种子（不依赖导出顺序）；
    - 每轨使用独立的 PCG64 生成器，互不消耗；
    - 并行（线程）与串行、正序与逆序的结果逐样本一致。

    ``tracks`` 支持：
    - dict：``{名称: 信号}`` 或 ``{名称: {"samples": ..., "id": ...}}``；
    - list/tuple：元素为信号或含 ``samples`` 的 dict；
    - 单个信号/标量：自动包装成一条轨道。

    :return: ``{轨道键: 处理报告}``，报告含 ``samples`` / ``float`` /
        ``metrics`` / ``seed`` 等字段。
    """
    # 整理成 [(key, identity, spec), ...]
    items = []
    if isinstance(tracks, dict):
        for index, (key, spec) in enumerate(tracks.items()):
            items.append((str(key), _track_identity(index, {"id": key, **({"samples": spec} if not isinstance(spec, dict) else spec)}), spec))
    elif isinstance(tracks, (list, tuple)) and tracks and all(
            not _is_number(t) for t in tracks):
        for index, spec in enumerate(tracks):
            identity = _track_identity(index, spec)
            items.append((_track_key(index, spec), identity, spec))
    else:
        spec = tracks
        items.append((_track_key(0, spec), _track_identity(0, spec), spec))

    def render_one(entry):
        key, identity, spec = entry
        if isinstance(spec, dict) and "samples" in spec and any(
                field in spec for field in ("sample_rate", "bits", "to_bits", "from_bits")):
            options = {
                "from_bits": spec.get("from_bits", from_bits),
                "to_bits": spec.get("to_bits", spec.get("bits", to_bits)),
                "sample_rate": spec.get("sample_rate", sample_rate),
                "segments": spec.get("segments", segments),
                "shape": spec.get("shape", shape),
                "dither": spec.get("dither", dither),
                "threshold_lsb": spec.get("threshold_lsb", threshold_lsb),
            }
            signal = spec["samples"]
            track_id = spec.get("id", identity)
        else:
            options = {
                "from_bits": from_bits, "to_bits": to_bits,
                "sample_rate": sample_rate, "segments": segments,
                "shape": shape, "dither": dither,
                "threshold_lsb": threshold_lsb,
            }
            signal = spec["samples"] if isinstance(spec, dict) and "samples" in spec else spec
            track_id = spec.get("id", identity) if isinstance(spec, dict) else identity

        result = requantize(signal, track=track_id, **options)
        result["metrics"] = metrics(result["float"], bits=result["bits"])
        result["track"] = key
        return key, result

    if parallel and len(items) > 1:
        workers = min(len(items), max(1, (os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rendered = list(pool.map(render_one, items))
    else:
        rendered = [render_one(entry) for entry in items]

    return {key: result for key, result in rendered}


# ---------------------------------------------------------------------------
# 校验 / 指标 / 审计 / 回退 / 确定性
# ---------------------------------------------------------------------------

def _safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _coeff_diagnostics(coeffs, sample_rate) -> dict:
    """系数的频响诊断：峰值位置/峰值、可听区与 18 kHz 以上的噪声增益。"""
    fs = _coerce_sample_rate(sample_rate)
    freqs, ntf = _shaper_response(coeffs, fs)
    nyquist = fs / 2.0
    audible = freqs <= min(13500.0, 0.55 * nyquist)
    above_18k = freqs >= SHAPER_TOP_HZ
    peak_index = int(np.argmax(ntf)) if ntf.size else 0
    return {
        "sample_rate": fs,
        "order": int(np.asarray(coeffs).size),
        "peak_gain": _safe_float(ntf[peak_index]) if ntf.size else 1.0,
        "peak_hz": _safe_float(freqs[peak_index]) if freqs.size else 0.0,
        "audible_max_gain": _safe_float(np.max(ntf[audible])) if np.any(audible) else 1.0,
        "audible_rms_gain": _safe_float(
            np.sqrt(np.mean(ntf[audible] ** 2))) if np.any(audible) else 1.0,
        "above_18k_max_gain": _safe_float(
            np.max(ntf[above_18k])) if np.any(above_18k) else 1.0,
        "stable": bool(np.max(ntf) < 8.0) if ntf.size else True,
    }


def validate_dither(samples=None, *, sample_rate=DEFAULT_SAMPLE_RATE,
                    bits=DEFAULT_TARGET_BITS, from_bits=DEFAULT_SOURCE_BITS,
                    track=None) -> dict:
    """自检抖动/整形实现的六条不变量，返回 ``{"ok", "checks"}``。

    检查项：
    1. 按轨种子：同轨稳定、异轨不同、与调用顺序无关；
    2. 系数按采样率：44.1 kHz 与 48 kHz 系数不同，且抬升带不越过
       18 kHz、奈奎斯特附近回落到单位增益；
    3. 静音跳过：全零输入输出逐样本为零（``skipped_silence`` 覆盖全部）；
    4. 跨段延续：分段处理与整段处理逐样本一致，段边界无额外阶跃；
    5. 抖动在限幅前：越界峰值被钳位且码值不越界（无回绕）；
    6. 峰值钳位：``clamp_peak`` 输出在对称满量程内。

    对任意输入（包括标量）都返回报告，不会抛出异常。
    """
    checks = {}

    # 1. 按轨种子
    seed_a = seed_per_track("A")
    seed_a2 = seed_per_track("A")
    seed_b = seed_per_track("B")
    seed_alt = seed_per_track({"id": "A"})
    checks["per_track_seed"] = {
        "ok": seed_a == seed_a2 and seed_a != seed_b and seed_a == seed_alt,
        "stable": seed_a == seed_a2,
        "distinct": seed_a != seed_b,
        "dict_key_normalized": seed_a == seed_alt,
        "seed": seed_a,
    }

    # 2. 系数按采样率
    fs_lo, fs_hi = 44100, 48000
    coeff_lo = coeff_by_rate(fs_lo)
    coeff_hi = coeff_by_rate(fs_hi)
    diag_lo = _coeff_diagnostics(coeff_lo, fs_lo)
    diag_hi = _coeff_diagnostics(coeff_hi, fs_hi)
    diag_96 = _coeff_diagnostics(coeff_by_rate(96000), 96000)
    differentiation = not np.allclose(coeff_lo, coeff_hi)
    # 抬升带（以及整形噪声本身）必须止步于 18 kHz 附近：
    # 高采样率下不能用按奈奎斯特比例铺开的系数，否则下变换时折叠回可听区
    top_ok = all(d["above_18k_max_gain"] <= 1.30
                 for d in (diag_lo, diag_hi, diag_96))
    checks["coeff_by_rate"] = {
        "ok": bool(differentiation and top_ok and diag_lo["stable"]
                   and diag_hi["stable"] and diag_96["stable"]),
        "coeffs_differ": bool(differentiation),
        "above_18k_bounded": bool(top_ok),
        "44100": diag_lo,
        "48000": diag_hi,
        "96000": diag_96,
    }

    # 3. 静音跳过
    silence = np.zeros(256, dtype=np.float64)
    silence_result = requantize(silence, from_bits=bits, to_bits=bits,
                                sample_rate=sample_rate, track="silence")
    all_zero = bool(np.all(silence_result["float"] == 0.0)
                    and np.all(silence_result["samples"] == 0))
    checks["silence_skip"] = {
        "ok": all_zero and silence_result["skipped_silence"] == silence.size,
        "samples": int(silence.size),
        "skipped_silence": silence_result["skipped_silence"],
        "output_is_zero": all_zero,
    }

    # 4. 跨段延续
    fs = _coerce_sample_rate(sample_rate)
    t = np.arange(1024, dtype=np.float64) / fs
    signal = 0.5 * np.sin(2.0 * np.pi * 997.0 * t)
    segmented = carry_state_across_segments(
        signal, segments=[[0, 300], [300, 700], [700, 1024]],
        sample_rate=fs, bits=bits, track="segment-test")
    within_range = bool(np.max(np.abs(segmented["output"])) <= 1.0)
    checks["carry_state"] = {
        "ok": bool(segmented["single_pass_equal"] and within_range),
        "single_pass_equal": segmented["single_pass_equal"],
        "segments": segmented["segments"],
        "seam_max_delta": segmented["seam_max_delta"],
        "within_full_scale": within_range,
    }

    # 5. 抖动在限幅前 / 6. 峰值钳位
    hot = np.array([1.5, -1.5, 1.0 - 1e-9, -1.0, 1.234567], dtype=np.float64)
    hot_result = requantize(hot, from_bits=bits, to_bits=bits,
                            sample_rate=sample_rate, track="peak-test")
    limit = _max_code(bits)
    codes_in_range = bool(np.all(hot_result["samples"] <= limit)
                          and np.all(hot_result["samples"] >= -limit))
    clamped = clamp_peak(hot, bits=bits)
    clamp_ok = bool(np.all(np.abs(clamped) <= _symmetric_limit(bits) + _EPS))
    checks["dither_before_limit"] = {
        "ok": bool(codes_in_range and clamp_ok),
        "pipeline_order": list(PIPELINE_ORDER),
        "codes_in_range": codes_in_range,
        "clipped": hot_result["clipped"],
        "clamp_ok": clamp_ok,
    }

    checks["clamp_peak"] = {
        "ok": clamp_ok,
        "symmetric_limit": _symmetric_limit(bits),
        "max_code": limit,
        "values": clamped.tolist(),
    }

    return {
        "ok": all(bool(check["ok"]) for check in checks.values()),
        "sample_rate": _coerce_sample_rate(sample_rate),
        "bits": _coerce_bits(bits),
        "checks": checks,
    }


def metrics(processed, reference=None, *, from_bits=None,
            bits=DEFAULT_TARGET_BITS) -> dict:
    """计算信号指标；给出参考信号时同时给出量化误差/SNR。

    返回 dict：``samples`` / ``peak`` / ``rms`` / ``silence_ratio`` /
    ``dc`` / ``max_abs``；若提供 ``reference``，另有 ``snr_db``、
    ``error_rms``、``error_max``、``error_rms_lsb``。
    """
    target_bits = _coerce_bits(bits)
    y = _as_signal(processed, from_bits=from_bits)
    report = {
        "samples": int(y.size),
        "peak": 0.0,
        "rms": 0.0,
        "silence_ratio": 1.0,
        "dc": 0.0,
        "max_abs": 0.0,
        "bits": target_bits,
    }
    if y.size == 0:
        return report

    report["peak"] = _safe_float(np.max(np.abs(y)))
    report["max_abs"] = report["peak"]
    report["rms"] = _safe_float(np.sqrt(np.mean(y ** 2)))
    report["dc"] = _safe_float(np.mean(y))
    report["silence_ratio"] = _safe_float(np.count_nonzero(y == 0.0) / y.size)

    if reference is not None:
        ref = _as_signal(reference, from_bits=from_bits)
        n = min(ref.size, y.size)
        if n > 0:
            ref, est = ref[:n], y[:n]
            error = ref - est
            signal_power = _safe_float(np.mean(ref ** 2))
            error_power = _safe_float(np.mean(error ** 2))
            snr = 10.0 * math.log10((signal_power + _EPS) / (error_power + _EPS))
            lsb = _quant_step(target_bits)
            report.update({
                "snr_db": _safe_float(snr),
                "error_rms": _safe_float(np.sqrt(error_power)),
                "error_max": _safe_float(np.max(np.abs(error))) if n else 0.0,
                "error_rms_lsb": _safe_float(np.sqrt(error_power) / lsb),
                "reference_samples": int(ref.size),
            })
    return report


def audit(record=None, **extra) -> dict:
    """生成确定性的处理审计记录（不含时间戳，便于 diff）。

    内容包括模块版本、管线顺序、默认位深/采样率、种子策略、常见采样率
    下的整形系数诊断。``record`` 为 dict 时并入，其余位置参数被忽略，
    保证 ``audit(1)`` 也能返回报告。
    """
    rates = (44100, 48000, 96000)
    entry = {
        "module": "bitdepth_dither",
        "version": 1,
        "pipeline_order": list(PIPELINE_ORDER),
        "defaults": {
            "source_bits": DEFAULT_SOURCE_BITS,
            "target_bits": DEFAULT_TARGET_BITS,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "shaper_order": SHAPER_ORDER,
        },
        "seed_policy": {
            "derivation": "blake2b-64(salt|track-identity)",
            "independent_per_track": True,
            "global_rng": False,
            "rng": "PCG64",
            "advance_by": "absolute sample index (2 uniforms/sample)",
        },
        "dither": {
            "type": "TPDF",
            "peak_lsb": 1.0,
            "silence": "skipped (digital silence stays zero)",
            "order": "before limit",
        },
        "shaper": {
            "top_hz": SHAPER_TOP_HZ,
            "top_ratio": SHAPER_TOP_RATIO,
            "audio_gain": SHAPER_AUDIO_GAIN,
            "mid_gain": SHAPER_MID_GAIN,
            "by_rate": {
                str(fs): _coeff_diagnostics(coeff_by_rate(fs), fs)
                for fs in rates
            },
        },
        "peak_clamp": {
            "mode": "symmetric full scale",
            "limit": "(2**(bits-1) - 1) / 2**(bits-1)",
        },
    }
    if isinstance(record, dict):
        entry["record"] = record
    elif record is not None:
        entry["record"] = {"input": repr(record)}
    entry.update(extra)
    return entry


def fallback(original, processed=None):
    """处理失败时的安全回退：返回原信号（直通），保证导出不中断。

    若 ``processed`` 为空、包含非有限值或无法转换，则回退到 ``original``。
    始终返回归一化 ndarray。
    """
    original_signal = _as_signal(original)
    if processed is None:
        return original_signal
    try:
        # 先检查原始数组的有限性：_as_signal 会把 NaN/Inf 清洗掉，
        # 清洗后再检查就永远发现不了坏数据
        raw = _raw_array(processed)
        if raw is None:
            return original_signal
        if raw.size == 0:
            return original_signal
        if np.issubdtype(raw.dtype, np.number) and not np.all(np.isfinite(
                raw.astype(np.float64, copy=False))):
            return original_signal
        candidate = _as_signal(raw)
        if candidate.size == 0:
            return original_signal
        if not np.all(np.isfinite(candidate)):
            return original_signal
        return candidate
    except Exception:
        return original_signal


def determinism(samples=None, *, track=None, sample_rate=DEFAULT_SAMPLE_RATE,
                bits=DEFAULT_TARGET_BITS, from_bits=DEFAULT_SOURCE_BITS,
                segments=None) -> dict:
    """验证两次导出逐样本一致（含分段/多轨独立性）。

    返回 dict：``equal``、``max_abs_diff``、``segmented_equal``、
    ``multi_track_independent``、``seed``、两个结果数组。
    """
    signal = _as_signal(samples if samples is not None else [0.0],
                        from_bits=from_bits)
    track_id = track if track is not None else "determinism"

    first = requantize(signal, from_bits=from_bits, to_bits=bits,
                       sample_rate=sample_rate, track=track_id,
                       segments=segments)
    second = requantize(signal, from_bits=from_bits, to_bits=bits,
                        sample_rate=sample_rate, track=track_id,
                        segments=segments)

    equal = bool(np.array_equal(first["samples"], second["samples"]))
    diff = (float(np.max(np.abs(first["float"] - second["float"])))
            if first["float"].size and second["float"].size else 0.0)

    segmented = carry_state_across_segments(
        signal, segments=segments if segments is not None else 4,
        sample_rate=sample_rate, bits=bits, track=track_id)
    segmented_equal = bool(segmented["single_pass_equal"] and equal)

    # 多轨独立性：同时导出 A/B 与单独导出 A 的结果必须一致
    packed = export({"A": signal, "B": signal * 0.5},
                    from_bits=from_bits, to_bits=bits,
                    sample_rate=sample_rate, parallel=True)
    alone = export({"A": signal}, from_bits=from_bits, to_bits=bits,
                   sample_rate=sample_rate, parallel=False)
    independent = bool(np.array_equal(packed["A"]["samples"],
                                      alone["A"]["samples"]))
    different_tracks = not np.array_equal(packed["A"]["samples"],
                                          packed["B"]["samples"])

    return {
        "equal": equal and independent,
        "max_abs_diff": diff,
        "segmented_equal": segmented_equal,
        "multi_track_independent": independent,
        "tracks_distinct": bool(different_tracks),
        "seed": int(first["seed"]),
        "first": first["samples"],
        "second": second["samples"],
        "sample_rate": first["sample_rate"],
        "bits": first["bits"],
    }
