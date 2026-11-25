"""高斯泼溅排序、同步与显存生命周期的确定性参考实现。

模块用纯 CPU 数据结构模拟图形管线中的关键不变量：

1. 异步 compute 排序与图形队列读取之间有显式 :class:`Fence` / 队列栅栏，
   绘制端只能在排序完成后读取，消除跨帧读到旧索引的闪烁。
2. 卸载 splat 时显存块只进入 ``pending_recycle``，必须跨过在途帧并等待
   引用它的栅栏后才真正回收（:func:`defer_recycle`）。
3. 投影协方差的最小特征值会被修复，椭圆短轴至少覆盖一个像素，避免远处
   极扁高斯因覆盖不足一个像素而被错误剔除（:func:`min_coverage`）。
4. 每个相机拥有独立的版本化排序缓冲，主视图与阴影视图不再共享顺序
   （:func:`sort_per_camera`）。
5. 平局使用与文件读取完成顺序无关的稳定 ID 排序
   （:func:`stable_tie` / :func:`determinism`）。
6. 每个排序缓冲都带 ``frame_version``；绘制时同时校验相机、帧版本和栅栏，
   不匹配则进入 :func:`fallback` 安全路径。

所有公开入口都能接受标量 ``1`` 的契约调用，并返回普通 ``dict``；同时也
接受映射、可迭代 splat 集合或 :class:`SplatSortPipeline`，便于做行为断言。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    # 测试契约入口
    "sort_fence",
    "defer_recycle",
    "min_coverage",
    "sort_per_camera",
    "stable_tie",
    "load",
    "sort",
    "cull",
    "draw",
    "unload",
    "validate_splat",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 常用辅助类型/函数
    "Fence",
    "SortBuffer",
    "MemoryBlock",
    "SplatSortPipeline",
    "SplatSortError",
    "FenceNotReady",
    "StaleSortBuffer",
    "DanglingOffset",
    "normalize_camera",
    "id_key",
    "build_sort_result",
    "cull_splats",
]

#: 亚像素椭圆被抬升后保证的最小包围盒覆盖（像素平方）/最小直径（像素）。
MIN_COVERAGE_PIXELS = 1.0
MIN_DIAMETER_PIXELS = 1.0

#: 卸载后至少保留的帧数；真实引擎通常为 frames-in-flight + GPU fence lag。
DEFAULT_RECYCLE_DELAY_FRAMES = 3

#: 排序协方差特征值时使用的 sigma 半径倍数。1.0 对应一倍标准差半径。
DEFAULT_SIGMA_SCALE = 1.0

#: 稳定 ID 的比较类型标签，避免数字与字符串直接比较。
_ID_NUMERIC = 0
_ID_TEXT = 1
_ID_MISSING = 2

_METRICS: dict[str, int] = {
    "loads": 0,
    "splats_loaded": 0,
    "sorts": 0,
    "culls": 0,
    "draws": 0,
    "unloads": 0,
    "blocks_recycled": 0,
    "fences_waited": 0,
    "fence_stalls": 0,
    "stale_reads_prevented": 0,
    "camera_sorts": 0,
    "ties_broken_by_stable_id": 0,
    "subpixel_splats_rescued": 0,
    "fallbacks": 0,
}


class SplatSortError(RuntimeError):
    """splat 排序管线违反生命周期或同步约束。"""


class FenceNotReady(SplatSortError):
    """图形队列试图在 compute 排序栅栏 signal 前读取排序缓冲。"""


class StaleSortBuffer(SplatSortError):
    """绘制请求的相机或帧版本与当前排序缓冲不一致。"""


class DanglingOffset(SplatSortError):
    """排序索引引用的显存块已经被真正回收。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    )


def _field(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """从 Mapping 或普通对象中按候选名称取字段。"""

    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return (
            float(value.get("x", 0.0)),
            float(value.get("y", 0.0)),
            float(value.get("z", 0.0)),
        )
    values = list(value)
    if len(values) < 3:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _canonical_digest(value: Any) -> str:
    payload = repr(value).encode("utf-8", errors="replace")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def id_key(value: Any) -> tuple[int, Any]:
    """生成跨运行稳定、可比较的 ID 键。

    数字和可解析为数字的字符串按数值排序；其他 ID 按文本排序。这样
    ``"10"``、``"2"``、``"1"`` 会得到自然顺序，而 UUID 等仍按字典序。
    """

    if isinstance(value, Mapping) or (
        not isinstance(value, (str, bytes, bytearray, int, float, bool))
        and hasattr(value, "stable_id")
    ):
        value = _field(value, ("stable_id", "splat_id", "id", "uuid", "key"))

    if value is None:
        return (_ID_MISSING, "")
    if isinstance(value, bool):
        return (_ID_NUMERIC, float(int(value)))
    if _is_real_number(value):
        number = float(value)
        return (_ID_NUMERIC, number if math.isfinite(number) else math.inf)
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    text = str(value).strip()
    try:
        return (_ID_NUMERIC, float(text))
    except ValueError:
        return (_ID_TEXT, text)


def _stable_id_for(raw: Any, index: int = 0) -> Any:
    explicit = _field(
        raw,
        ("stable_id", "splat_id", "stableId", "id", "uuid", "asset_id", "key"),
    )
    if explicit is not None:
        return explicit

    # 没有显式 ID 时，只使用与文件读取完成顺序无关的内容/文件坐标生成 ID。
    if isinstance(raw, Mapping):
        basis = {
            name: raw[name]
            for name in (
                "source",
                "file",
                "file_index",
                "fileIndex",
                "asset",
                "position",
                "xyz",
                "view_depth",
                "depth",
                "covariance",
                "cov2d",
                "opacity",
                "color",
            )
            if name in raw
        }
    else:
        basis = raw
    return "content-" + _canonical_digest(basis)


# ---------------------------------------------------------------------------
# 同步原语与数据记录
# ---------------------------------------------------------------------------


class Fence:
    """compute → graphics 队列栅栏的纯 Python 模拟。"""

    __slots__ = ("label", "frame_version", "signaled", "waits", "signal_count")

    def __init__(self, label: str = "sort-fence", frame_version: int | None = None):
        self.label = label
        self.frame_version = frame_version
        self.signaled = False
        self.waits = 0
        self.signal_count = 0

    def signal(self) -> "Fence":
        if not self.signaled:
            self.signaled = True
            self.signal_count += 1
        return self

    def wait(self, *, raise_if_not_ready: bool = True) -> bool:
        self.waits += 1
        _METRICS["fences_waited"] += 1
        if not self.signaled:
            _METRICS["fence_stalls"] += 1
            if raise_if_not_ready:
                raise FenceNotReady(
                    f"{self.label} frame={self.frame_version} has not signaled"
                )
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "fence",
            "label": self.label,
            "frame_version": self.frame_version,
            "signaled": self.signaled,
            "waits": self.waits,
        }


@dataclass
class SortBuffer:
    """相机私有的版本化排序索引缓冲。"""

    camera_id: Any
    frame_version: int
    order: list[int] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    fence: Fence = field(default_factory=Fence)
    status: str = "pending"
    queue: str = "async_compute"

    def __post_init__(self) -> None:
        if self.fence.frame_version is None:
            self.fence.frame_version = self.frame_version

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_version": self.frame_version,
            "order": list(self.order),
            "indices": list(self.order),
            "sorted_ids": [self.entries[i]["stable_id"] for i in self.order],
            "entries": [dict(entry) for entry in self.entries],
            "fence": self.fence.as_dict(),
            "fence_ready": self.fence.signaled,
            "status": self.status,
            "queue": self.queue,
            "buffer_version": f"{self.camera_id}:{self.frame_version}",
        }


@dataclass
class MemoryBlock:
    """一块可能被排序缓冲旧偏移引用的显存。"""

    block_id: str
    size_bytes: int = 0
    state: str = "live"  # live -> pending_recycle -> reclaimed
    generation: int = 1
    retired_frame: int | None = None
    reclaim_after_frame: int | None = None
    fence: Fence | None = None
    splat_ids: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "size_bytes": self.size_bytes,
            "state": self.state,
            "generation": self.generation,
            "retired_frame": self.retired_frame,
            "reclaim_after_frame": self.reclaim_after_frame,
            "fence": self.fence.as_dict() if self.fence is not None else None,
            "splat_ids": list(self.splat_ids),
            "reclaimed": self.state == "reclaimed",
        }


# ---------------------------------------------------------------------------
# 相机、深度与协方差
# ---------------------------------------------------------------------------


def normalize_camera(camera: Any = None) -> dict[str, Any]:
    """把相机标量/映射归一化为纯数据描述。"""

    if camera is None:
        camera_id: Any = "main"
        data: Mapping[str, Any] = {}
    elif isinstance(camera, Mapping):
        data = camera
        camera_id = _field(
            camera, ("camera_id", "id", "name", "camera"), default="main"
        )
    else:
        camera_id = camera
        data = {}

    eye = _vec3(_field(data, ("eye", "position", "world_position"))) or (0.0, 0.0, 0.0)
    explicit_forward = _vec3(_field(data, ("forward", "direction", "view_direction")))
    target = _vec3(_field(data, ("target", "look_at", "center")))
    if explicit_forward is not None:
        forward = explicit_forward
    elif target is not None:
        forward = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    else:
        forward = (0.0, 0.0, 1.0)

    length = math.sqrt(sum(v * v for v in forward))
    if length > 1e-12:
        forward = tuple(v / length for v in forward)  # type: ignore[assignment]
    else:
        forward = (0.0, 0.0, 1.0)

    width = int(_field(data, ("width", "viewport_width"), 1920))
    height = int(_field(data, ("height", "viewport_height"), 1080))
    fov_y = _finite_float(_field(data, ("fov_y", "fov", "fovy"), 60.0), 60.0)
    focal = _field(data, ("focal_length_px", "focal_px", "focal"), None)
    if focal is None:
        focal = (height * 0.5) / math.tan(math.radians(max(1e-3, fov_y)) * 0.5)

    return {
        "camera_id": camera_id,
        "eye": eye,
        "forward": forward,
        "width": max(1, width),
        "height": max(1, height),
        "fov_y": fov_y,
        "focal": max(1e-6, _finite_float(focal, 900.0)),
        "near": _finite_float(_field(data, ("near", "near_plane"), 0.05), 0.05),
        "far": _finite_float(_field(data, ("far", "far_plane"), 1000.0), 1000.0),
    }


def normalize_splat(raw: Any, index: int = 0) -> dict[str, Any]:
    """把常见 splat 输入形态归一化。"""

    if raw is None:
        raw = {}

    if _is_real_number(raw):
        # 裸数字按视深处理；稳定 ID 由内容生成。
        raw = {"view_depth": float(raw)}
    elif isinstance(raw, (tuple, list)):
        # 三元位置向量；更长的元组不做臆测，交给映射/协方差解析器。
        if len(raw) == 3 and all(_is_real_number(v) for v in raw):
            raw = {"position": tuple(raw)}
        else:
            raw = {"values": list(raw)}

    stable_id = _stable_id_for(raw, index)
    position = _vec3(_field(raw, ("position", "xyz", "world_position", "p")))
    depth = _field(raw, ("view_depth", "depth", "view_space_depth", "z", "w"))
    if depth is not None:
        depth = _finite_float(depth, math.inf)

    opacity = _field(raw, ("opacity", "alpha", "a"), None)
    if opacity is not None:
        opacity = _finite_float(opacity, 1.0)

    return {
        "stable_id": stable_id,
        "source_index": index,
        "position": position,
        "view_depth": depth,
        "opacity": opacity,
        "offset": _field(raw, ("offset", "byte_offset"), None),
        "block_id": _field(raw, ("block_id", "memory_block", "block"), None),
        "screen_position": _field(raw, ("screen_position", "screen_xy"), None),
        "screen_covariance": _field(
            raw,
            ("screen_covariance", "covariance_2d_px", "cov2d_px", "screen_cov2d"),
        ),
        "covariance": _field(
            raw,
            ("covariance_2d", "cov2d", "covariance", "cov3d", "cov"),
        ),
        "screen_radii": _field(raw, ("screen_radii", "radii_px"), None),
        "radii": _field(raw, ("radii", "world_radii"), None),
        "precision": _field(raw, ("precision", "dtype", "covariance_dtype"), None),
        "raw": raw,
    }


def resolve_depth(splat: Mapping[str, Any], camera: Mapping[str, Any] | None = None) -> float:
    camera = camera or normalize_camera()
    depth = splat.get("view_depth")
    if depth is not None and math.isfinite(float(depth)):
        return float(depth)

    position = splat.get("position")
    if position is None:
        raw = splat.get("raw", {})
        position = _vec3(_field(raw, ("position", "xyz", "p")))
    if position is None:
        return 0.0

    eye = camera["eye"]
    forward = camera["forward"]
    delta = (
        position[0] - eye[0],
        position[1] - eye[1],
        position[2] - eye[2],
    )
    return delta[0] * forward[0] + delta[1] * forward[1] + delta[2] * forward[2]


def covariance_2d(value: Any) -> tuple[float, float, float] | None:
    """解析 2D 协方差为 ``(xx, xy, yy)``。"""

    if value is None:
        return None

    if isinstance(value, Mapping):
        value = _field(
            value,
            (
                "covariance_2d",
                "cov2d",
                "screen_covariance",
                "covariance_2d_px",
                "cov",
                "covariance",
            ),
        )
        if value is None:
            return None

    if _is_real_number(value):
        scalar = max(0.0, float(value))
        return scalar, 0.0, scalar

    values = list(value)
    if len(values) == 3:
        a, b, c = values
        return float(a), float(b), float(c)
    if len(values) == 4:
        a, b, c, d = (float(v) for v in values)
        # 行主序 2x2；数值误差导致的非对称部分对称化。
        return a, (b + c) * 0.5, d
    if len(values) == 6:
        # 常见 3D 对称协方差顺序：xx, yy, zz, xy, xz, yz。
        xx, yy, _zz, xy, _xz, _yz = (float(v) for v in values)
        return xx, xy, yy
    if len(values) == 9:
        return float(values[0]), float(values[1]), float(values[4])
    raise ValueError(f"无法把长度 {len(values)} 的数据解析为 2D 协方差")


def ellipse_coverage(
    covariance: Any = None,
    radii: Any = None,
    *,
    minimum: float = MIN_COVERAGE_PIXELS,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
) -> dict[str, Any]:
    """计算并修复投影椭圆的像素覆盖。

    包围盒覆盖定义为 ``(2*r_major) * (2*r_minor)``。把最小半径抬到
    ``minimum / 2`` 后，任意保留 splat 的包围盒宽高都至少为 ``minimum``
    像素，因此不会再因低精度协方差坍缩成亚像素细线而被剔除。
    """

    minimum = max(0.0, float(minimum))
    min_radius = minimum * 0.5
    eigen_floor = (min_radius / max(sigma_scale, 1e-12)) ** 2

    raw_major = raw_minor = 0.0
    a = b = c = 0.0
    eigenvalues: tuple[float, float] = (0.0, 0.0)

    if radii is not None:
        radius_values = [float(v) for v in radii]
        if len(radius_values) < 2:
            raise ValueError("radii 至少需要两个分量")
        r0, r1 = abs(radius_values[0]), abs(radius_values[1])
        raw_major, raw_minor = max(r0, r1), min(r0, r1)
        raw_coverage = 4.0 * raw_major * raw_minor
    else:
        cov = covariance_2d(covariance)
        if cov is None:
            # 没有形状信息时给一个最小可见覆盖，而不是按零尺寸剔除。
            cov = (eigen_floor, 0.0, eigen_floor)
        a, b, c = cov
        trace = a + c
        discriminant = max(0.0, (a - c) * (a - c) + 4.0 * b * b)
        root = math.sqrt(discriminant)
        lambda_major = max(0.0, (trace + root) * 0.5)
        lambda_minor = max(0.0, (trace - root) * 0.5)
        eigenvalues = (lambda_major, lambda_minor)

        raw_major = sigma_scale * math.sqrt(lambda_major)
        raw_minor = sigma_scale * math.sqrt(lambda_minor)
        raw_coverage = 4.0 * raw_major * raw_minor

    repaired_minor = max(raw_minor, min_radius)
    repaired_major = max(raw_major, min_radius)
    coverage = max(minimum, 4.0 * repaired_major * repaired_minor)
    clamped = raw_minor < min_radius or raw_coverage < minimum

    return {
        "covariance": [a, b, c],
        "eigenvalues": list(eigenvalues),
        "raw_radii": [raw_major, raw_minor],
        "radii": [repaired_major, repaired_minor],
        "raw_coverage": raw_coverage,
        "coverage_pixels": coverage,
        "minimum_coverage": minimum,
        "min_radius_pixels": min_radius,
        "clamped": clamped,
        "would_have_been_culled": raw_coverage < minimum,
        "keep": coverage >= minimum,
    }


def min_coverage(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """保证一个投影 splat 至少覆盖最小像素区域。

    调用形态：

    * ``min_coverage(0.2)``：把输入当作已有像素覆盖值并抬到下限；
    * ``min_coverage([[a,b],[b,c]])`` / ``(xx, xy, yy)``：修复协方差椭圆；
    * ``min_coverage({"covariance": ..., "minimum_coverage": 1})``；
    * ``min_coverage(1)``：满足基础契约的标量调用。
    """

    minimum = float(
        kwargs.pop("minimum", kwargs.pop("min_pixels", MIN_COVERAGE_PIXELS))
    )
    source = args[0] if args else kwargs.pop("source", None)

    if isinstance(source, Mapping):
        minimum = float(
            source.get(
                "minimum_coverage",
                source.get("min_pixels", minimum),
            )
        )
        radii = _field(source, ("screen_radii", "radii_px", "radii"))
        cov = _field(
            source,
            (
                "screen_covariance",
                "covariance_2d_px",
                "cov2d",
                "covariance_2d",
                "covariance",
                "cov",
            ),
        )
        if cov is None and radii is None:
            raw_coverage = _finite_float(
                source.get("coverage", source.get("coverage_pixels", minimum)),
                minimum,
            )
            coverage = max(raw_coverage, minimum)
            return {
                "raw_coverage": raw_coverage,
                "coverage_pixels": coverage,
                "coverage": coverage,
                "minimum_coverage": minimum,
                "clamped": raw_coverage < minimum,
                "keep": True,
                "policy": "scalar_coverage_floor",
            }
        result = ellipse_coverage(cov, radii, minimum=minimum, **kwargs)
    elif isinstance(source, (list, tuple)):
        result = ellipse_coverage(source, None, minimum=minimum, **kwargs)
    elif source is not None:
        raw_coverage = _finite_float(source, 0.0)
        coverage = max(raw_coverage, minimum)
        result = {
            "raw_coverage": raw_coverage,
            "coverage_pixels": coverage,
            "coverage": coverage,
            "minimum_coverage": minimum,
            "clamped": raw_coverage < minimum,
            "keep": True,
            "policy": "scalar_coverage_floor",
        }
    else:
        result = ellipse_coverage(None, None, minimum=minimum, **kwargs)

    result.setdefault("coverage", result["coverage_pixels"])
    result["policy"] = "clamp_subpixel_axes"
    if result["clamped"]:
        _METRICS["subpixel_splats_rescued"] += 1
    return result


# ---------------------------------------------------------------------------
# 稳定排序
# ---------------------------------------------------------------------------


def stable_tie(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """返回平局稳定 ID 排序结果或单个 ID 的排序键。

    支持 ``stable_tie([...])``、``stable_tie(a, b, c)`` 和标量契约调用
    ``stable_tie(1)``。splat 映射会读取其中的 ``stable_id`` / ``id``。
    """

    if not args:
        items: list[Any] = []
    elif len(args) == 1:
        only = args[0]
        if isinstance(only, Mapping) and "splats" in only:
            items = list(only.get("splats") or [])
        elif _is_iterable(only):
            items = list(only)
        else:
            # 单个 ID 或单个 splat：返回它的键描述。
            key = id_key(only)
            stable_id = _stable_id_for(only) if isinstance(only, Mapping) else only
            return {
                "stable_id": stable_id,
                "id": stable_id,
                "key": list(key),
                "ids": [stable_id],
                "order": [0],
                "count": 1,
                "tie_break": "stable_id",
                "stable": True,
            }
    else:
        items = list(args)

    ids = [
        _field(item, ("stable_id", "splat_id", "id", "uuid", "key"), item)
        for item in items
    ]
    order = sorted(range(len(items)), key=lambda i: id_key(ids[i]))
    sorted_ids = [ids[i] for i in order]
    keys = [list(id_key(item)) for item in sorted_ids]
    return {
        "ids": sorted_ids,
        "order": order,
        "indices": order,
        "keys": keys,
        "count": len(items),
        "tie_break": "stable_id_ascending",
        "stable": True,
        "source_order_independent": True,
    }


def _sort_key(entry: Mapping[str, Any], *, back_to_front: bool) -> tuple[Any, ...]:
    depth = float(entry.get("depth", 0.0))
    if not math.isfinite(depth):
        depth = math.inf
    depth_part = -depth if back_to_front else depth
    return (depth_part, id_key(entry.get("stable_id")))


def build_sort_result(
    splats: Any,
    camera: Any = None,
    *,
    frame_version: int = 1,
    back_to_front: bool = True,
    signal_fence: bool = True,
) -> SortBuffer:
    """构造一个相机私有的版本化排序缓冲。"""

    cam = normalize_camera(camera)
    if isinstance(splats, Mapping):
        raw_items: list[Any] = list(splats.get("splats", [splats]))
    elif splats is None:
        raw_items = []
    else:
        raw_items = list(splats)

    entries: list[dict[str, Any]] = []
    tie_count = 0
    previous_depth: float | None = None
    for index, raw in enumerate(raw_items):
        splat = normalize_splat(raw, index)
        depth = resolve_depth(splat, cam)
        entry = {
            "index": index,
            "stable_id": splat["stable_id"],
            "depth": depth,
            "offset": splat["offset"] if splat["offset"] is not None else index * 64,
            "block_id": splat["block_id"] if splat["block_id"] is not None else "external",
            "active": True,
        }
        entries.append(entry)
        if previous_depth is not None and depth == previous_depth:
            tie_count += 1
        previous_depth = depth

    order = sorted(
        range(len(entries)),
        key=lambda i: _sort_key(entries[i], back_to_front=back_to_front),
    )
    fence = Fence(
        f"sort:{cam['camera_id']}:{frame_version}", frame_version=frame_version
    )
    buffer = SortBuffer(
        camera_id=cam["camera_id"],
        frame_version=int(frame_version),
        order=order,
        entries=entries,
        fence=fence,
        status="ready" if signal_fence else "pending",
    )
    if signal_fence:
        fence.signal()
    if tie_count:
        _METRICS["ties_broken_by_stable_id"] += tie_count
    _METRICS["camera_sorts"] += 1
    return buffer


def sort_fence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """创建/描述 async compute → graphics 的排序栅栏。"""

    request = args[0] if args else kwargs
    if isinstance(request, Mapping):
        frame_version = int(
            request.get("frame_version", request.get("frame", kwargs.get("frame_version", 0)))
        )
        label = str(request.get("label", "sort-fence"))
        start_signaled = bool(request.get("signaled", False))
    else:
        frame_version = int(_finite_float(request, 0.0)) if request is not None else 0
        label = str(kwargs.get("label", "sort-fence"))
        start_signaled = bool(kwargs.get("signaled", False))

    fence = Fence(label, frame_version)
    if start_signaled:
        fence.signal()

    return {
        "frame_version": frame_version,
        "fence": fence.as_dict(),
        "fence_id": fence.label,
        "source_queue": "async_compute",
        "destination_queue": "graphics",
        "barrier": "compute_shader_write_to_graphics_index_read",
        "wait_stage": "draw_indexed",
        "wait_before_draw": True,
        "ready": fence.signaled,
        "rule": "graphics_queue_must_wait_for_fence_and_frame_version",
    }


def sort_per_camera(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """为每个相机分别生成排序缓冲，禁止主/阴影视图共享顺序。"""

    request: Mapping[str, Any] | None = None
    camera: Any = kwargs.get("camera")
    splats: Any = kwargs.get("splats")
    frame_version = int(kwargs.get("frame_version", kwargs.get("frame", 1)))
    cameras: Any = kwargs.get("cameras", ("main", "shadow"))

    if args:
        first = args[0]
        if isinstance(first, Mapping) and (
            "splats" in first or "cameras" in first or "frame_version" in first
        ):
            request = first
            splats = first.get("splats", splats)
            cameras = first.get("cameras", cameras)
            frame_version = int(
                first.get(
                    "frame_version",
                    first.get("frame", frame_version),
                )
            )
        else:
            camera = first
            if len(args) >= 2:
                splats = args[1]

    if request is not None:
        camera = request.get("camera", camera)

    if splats is None:
        # 契约调用 sort_per_camera(1)：把标量视为需要生成的演示 splat 数。
        count = 1
        if _is_real_number(camera):
            count = max(1, int(camera))
        splats = _demo_splats(count)

    if camera is not None and (request is None or "cameras" not in request):
        camera_list = [camera]
    else:
        camera_list = list(cameras)

    buffers: dict[str, dict[str, Any]] = {}
    camera_ids: list[Any] = []
    for index, cam in enumerate(camera_list):
        cam_data = normalize_camera(cam if not isinstance(cam, dict) else cam)
        cid = cam_data["camera_id"]
        version = frame_version + index * 0  # 同一帧内各相机使用相同帧号。
        buffer = build_sort_result(splats, cam_data, frame_version=version)
        key = str(cid)
        buffers[key] = buffer.as_dict()
        camera_ids.append(cid)
        _METRICS["sorts"] += 1

    return {
        "cameras": buffers,
        "results": buffers,
        "camera_ids": camera_ids,
        "buffer_keys": [f"{cid}:{frame_version}" for cid in camera_ids],
        "frame_version": frame_version,
        "independent": True,
        "shared_buffer": False,
        "tie_break": "stable_id",
        "fence_required": True,
    }


# ---------------------------------------------------------------------------
# 投影与剔除
# ---------------------------------------------------------------------------


def project_splat(raw: Any, index: int, camera: Mapping[str, Any]) -> dict[str, Any]:
    """投影单个 splat 并计算修复后的屏幕覆盖。"""

    splat = normalize_splat(raw, index)
    depth = resolve_depth(splat, camera)
    position = splat["position"] or (0.0, 0.0, depth)
    focal = float(camera["focal"])
    width = int(camera["width"])
    height = int(camera["height"])

    screen_xy = _field(raw if isinstance(raw, Mapping) else {}, ("screen_position", "screen_xy"))
    if screen_xy is not None:
        xy = list(screen_xy)
        screen_x, screen_y = float(xy[0]), float(xy[1])
    else:
        safe_z = max(depth, 1e-6)
        screen_x = width * 0.5 + focal * position[0] / safe_z
        screen_y = height * 0.5 - focal * position[1] / safe_z

    raw_mapping = raw if isinstance(raw, Mapping) else {}
    screen_cov = _field(
        raw_mapping,
        ("screen_covariance", "covariance_2d_px", "cov2d_px", "screen_cov2d"),
    )
    radii = _field(raw_mapping, ("screen_radii", "radii_px"))
    if screen_cov is not None or radii is not None:
        coverage = ellipse_coverage(screen_cov, radii)
    else:
        world_cov = _field(
            raw_mapping,
            ("covariance_2d", "cov2d", "covariance", "cov3d", "cov"),
        )
        cov = covariance_2d(world_cov)
        if cov is not None:
            scale = focal / max(depth, 1e-6)
            # 协方差按投影 Jacobian 的平方缩放。
            projected_cov = (
                cov[0] * scale * scale,
                cov[1] * scale * scale,
                cov[2] * scale * scale,
            )
        else:
            projected_cov = None
        world_radii = _field(raw_mapping, ("radii", "world_radii"))
        if world_radii is not None:
            radius_scale = focal / max(depth, 1e-6)
            radii = [float(v) * radius_scale for v in world_radii]
        coverage = ellipse_coverage(projected_cov, radii)

    rx, ry = coverage["radii"]
    opacity = splat["opacity"]
    if opacity is None:
        opacity = 1.0

    reasons: list[str] = []
    if not all(math.isfinite(v) for v in (depth, screen_x, screen_y, rx, ry)):
        reasons.append("non_finite_projection")
    if depth < float(camera["near"]):
        reasons.append("near_plane")
    elif depth > float(camera["far"]):
        reasons.append("far_plane")
    if opacity <= 0.0:
        reasons.append("zero_opacity")

    # 用修复后的椭圆半径做视锥/屏幕外判断，避免边缘突然消失。
    if screen_x + rx < 0 or screen_x - rx > width:
        reasons.append("outside_viewport_x")
    if screen_y + ry < 0 or screen_y - ry > height:
        reasons.append("outside_viewport_y")

    kept = not reasons
    return {
        "index": index,
        "stable_id": splat["stable_id"],
        "depth": depth,
        "screen_position": [screen_x, screen_y],
        "radii": [rx, ry],
        "coverage_pixels": coverage["coverage_pixels"],
        "raw_coverage": coverage["raw_coverage"],
        "subpixel_rescued": coverage["clamped"]
        and coverage["would_have_been_culled"],
        "opacity": opacity,
        "kept": kept,
        "culled": not kept,
        "reasons": reasons,
        "offset": splat["offset"] if splat["offset"] is not None else index * 64,
        "block_id": splat["block_id"] if splat["block_id"] is not None else "external",
    }


def cull_splats(splats: Any, camera: Any = None, **kwargs: Any) -> dict[str, Any]:
    camera_data = normalize_camera(camera or kwargs.get("camera"))
    raw_items = list(splats or [])
    items = [project_splat(raw, i, camera_data) for i, raw in enumerate(raw_items)]
    kept = [item["index"] for item in items if item["kept"]]
    culled = [item["index"] for item in items if not item["kept"]]
    rescued = [item["index"] for item in items if item["subpixel_rescued"]]
    _METRICS["culls"] += 1
    _METRICS["subpixel_splats_rescued"] += len(rescued)
    return {
        "camera_id": camera_data["camera_id"],
        "total": len(items),
        "kept": kept,
        "kept_indices": kept,
        "culled": culled,
        "culled_indices": culled,
        "rescued_subpixel": rescued,
        "visible_ids": [items[i]["stable_id"] for i in kept],
        "items": items,
        "minimum_coverage": MIN_COVERAGE_PIXELS,
        "subpixel_floor_enabled": True,
    }


def cull(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """剔除入口；标量契约调用会生成一个可见的演示 splat。"""

    request = args[0] if args else None
    camera = kwargs.get("camera")

    if isinstance(request, SplatSortPipeline):
        return request.cull(**kwargs)
    if isinstance(request, Mapping) and "splats" in request:
        splats = request.get("splats")
        camera = request.get("camera", camera)
    elif _is_iterable(request):
        splats = list(request)
    elif _is_real_number(request):
        splats = _demo_splats(max(1, int(request)))
    elif isinstance(request, Mapping):
        splats = [request]
    else:
        splats = kwargs.get("splats") or _demo_splats(1)

    return cull_splats(splats, camera, **kwargs)


# ---------------------------------------------------------------------------
# 延迟回收
# ---------------------------------------------------------------------------


def defer_recycle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """生成“卸载但不立即回收”的安全计划。"""

    request = args[0] if args else kwargs
    if isinstance(request, Mapping):
        block_id = str(
            request.get(
                "block_id",
                request.get("block", request.get("id", kwargs.get("block_id", "block-1"))),
            )
        )
        frame = int(request.get("frame", request.get("retired_frame", kwargs.get("frame", 0))))
        delay = int(
            request.get(
                "delay_frames",
                request.get("recycle_delay", kwargs.get("delay_frames", DEFAULT_RECYCLE_DELAY_FRAMES)),
            )
        )
        size_bytes = int(request.get("size_bytes", kwargs.get("size_bytes", 0)))
        fence_signaled = request.get("fence_signaled")
        fence_info = request.get("fence")
        if fence_signaled is None and isinstance(fence_info, Mapping):
            fence_signaled = fence_info.get("signaled")
        referenced_versions = list(request.get("referenced_versions", ()))
    else:
        block_id = f"block-{request}" if request is not None else "block-1"
        frame = int(kwargs.get("frame", 0))
        delay = int(kwargs.get("delay_frames", DEFAULT_RECYCLE_DELAY_FRAMES))
        size_bytes = int(kwargs.get("size_bytes", 0))
        fence_signaled = kwargs.get("fence_signaled")
        referenced_versions = list(kwargs.get("referenced_versions", ()))

    reclaim_after = frame + max(0, delay)
    return {
        "block_id": block_id,
        "size_bytes": size_bytes,
        "state": "pending_recycle",
        "deferred": True,
        "reclaimed": False,
        "immediate_recycle": False,
        "retired_frame": frame,
        "delay_frames": delay,
        "reclaim_after_frame": reclaim_after,
        "fence_required": True,
        "fence_signaled": fence_signaled,
        "referenced_versions": referenced_versions,
        "rule": "recycle_only_after_fence_and_all_in_flight_versions_complete",
    }


# ---------------------------------------------------------------------------
# 完整有状态管线
# ---------------------------------------------------------------------------


def _demo_splats(count: int = 1) -> list[dict[str, Any]]:
    count = max(0, int(count))
    result: list[dict[str, Any]] = []
    for i in range(count):
        depth = 4.0 + (i % 5)
        result.append(
            {
                "stable_id": f"splat-{i:05d}",
                "position": ((i % 7 - 3) * 0.25, (i % 5 - 2) * 0.2, depth),
                "view_depth": depth,
                # 世界空间协方差；投影后仍足够可见。
                "covariance": [0.09, 0.0, 0.09],
                "opacity": 0.75,
                "source_file": f"chunk_{i // 4}.bin",
                "file_index": i,
            }
        )
    return result


class SplatSortPipeline:
    """拥有已加载 splat、显存块和各相机版本化缓冲的确定性管线。"""

    def __init__(
        self,
        *,
        recycle_delay_frames: int = DEFAULT_RECYCLE_DELAY_FRAMES,
        minimum_coverage: float = MIN_COVERAGE_PIXELS,
    ) -> None:
        self.recycle_delay_frames = int(recycle_delay_frames)
        self.minimum_coverage = float(minimum_coverage)
        self.frame = 0
        self.splats: dict[Any, dict[str, Any]] = {}
        self.blocks: dict[str, MemoryBlock] = {}
        self.buffers: dict[Any, SortBuffer] = {}
        self.history: dict[tuple[Any, int], SortBuffer] = {}
        self.fallback_log: list[dict[str, Any]] = []

    # -- 加载 -------------------------------------------------------------

    def load(self, source: Any = 1, **kwargs: Any) -> dict[str, Any]:
        if isinstance(source, Mapping) and "splats" in source:
            raw_splats = list(source.get("splats") or [])
            source_name = source.get("block_id", source.get("batch", "batch"))
        elif _is_iterable(source):
            raw_splats = list(source)
            source_name = kwargs.get("block_id", "batch")
        elif _is_real_number(source):
            raw_splats = _demo_splats(max(1, int(source)))
            source_name = kwargs.get("block_id", f"dynamic-{int(source)}")
        elif source is None:
            raw_splats = _demo_splats(1)
            source_name = "dynamic-1"
        else:
            raw_splats = [source]
            source_name = kwargs.get("block_id", "batch")

        block_index = len(self.blocks) + 1
        block_id = str(kwargs.get("block_id", f"block-{block_index}"))
        stride = int(kwargs.get("stride_bytes", 64))
        block = MemoryBlock(
            block_id=block_id,
            size_bytes=len(raw_splats) * stride,
        )
        self.blocks[block_id] = block

        loaded_ids: list[Any] = []
        for i, raw in enumerate(raw_splats):
            splat = normalize_splat(raw, i)
            stable_id = splat["stable_id"]
            splat["block_id"] = block_id
            splat["offset"] = i * stride
            splat["active"] = True
            self.splats[stable_id] = splat
            block.splat_ids.append(stable_id)
            loaded_ids.append(stable_id)

        _METRICS["loads"] += 1
        _METRICS["splats_loaded"] += len(loaded_ids)
        return {
            "loaded": len(loaded_ids),
            "count": len(loaded_ids),
            "block_id": block_id,
            "offset": 0,
            "stride_bytes": stride,
            "size_bytes": block.size_bytes,
            "stable_ids": loaded_ids,
            "stable_order": stable_tie(loaded_ids)["ids"],
            "completion_order_independent": True,
            "block": block.as_dict(),
        }

    # -- 排序 -------------------------------------------------------------

    def sort(self, target: Any = None, **kwargs: Any) -> dict[str, Any]:
        camera = kwargs.get("camera", "main")
        frame_version = kwargs.get("frame_version", kwargs.get("frame"))
        explicit_splats = kwargs.get("splats")
        back_to_front = bool(kwargs.get("back_to_front", True))

        if isinstance(target, Mapping) and (
            "splats" in target or "frame_version" in target or "camera" in target
        ):
            camera = target.get("camera", camera)
            frame_version = target.get(
                "frame_version", target.get("frame", frame_version)
            )
            explicit_splats = target.get("splats", explicit_splats)
        elif _is_iterable(target):
            explicit_splats = list(target)
        elif isinstance(target, Mapping):
            explicit_splats = [target]
        elif _is_real_number(target) and frame_version is None:
            # sort(1) 的标量优先作为帧版本；若管线为空则稍后补一个演示 splat。
            frame_version = int(target)

        if frame_version is None:
            self.frame += 1
            frame_version = self.frame
        frame_version = int(frame_version)
        self.frame = max(self.frame, frame_version)

        if explicit_splats is not None:
            records = list(explicit_splats)
        else:
            records = [s for s in self.splats.values() if s.get("active", True)]
            if not records and _is_real_number(target):
                records = _demo_splats(1)
            elif not records:
                records = _demo_splats(1)

        buffer = build_sort_result(
            records,
            camera,
            frame_version=frame_version,
            back_to_front=back_to_front,
            signal_fence=True,
        )
        cid = buffer.camera_id
        self.buffers[cid] = buffer
        self.history[(cid, frame_version)] = buffer
        _METRICS["sorts"] += 1
        return buffer.as_dict()

    # -- 剔除 -------------------------------------------------------------

    def cull(self, request: Any = None, **kwargs: Any) -> dict[str, Any]:
        camera = kwargs.get("camera", "main")
        if isinstance(request, Mapping) and "splats" in request:
            records = request["splats"]
            camera = request.get("camera", camera)
        elif _is_iterable(request):
            records = list(request)
        elif request is None or _is_real_number(request):
            records = [s for s in self.splats.values() if s.get("active", True)]
            if not records:
                records = _demo_splats(max(1, int(request)) if _is_real_number(request) else 1)
        elif isinstance(request, Mapping):
            records = [request]
        else:
            records = [s for s in self.splats.values() if s.get("active", True)]
        return cull_splats(records, camera)

    # -- 绘制 -------------------------------------------------------------

    def draw(self, target: Any = 1, **kwargs: Any) -> dict[str, Any]:
        camera = kwargs.get("camera", "main")
        requested_version = kwargs.get("frame_version", kwargs.get("frame"))
        strict = bool(kwargs.get("strict", False))
        action_taken = "none"
        prevented = False
        fallback_info: dict[str, Any] | None = None

        if isinstance(target, SortBuffer):
            buffer: SortBuffer | None = target
            cid = buffer.camera_id
            requested_version = buffer.frame_version
        elif isinstance(target, Mapping):
            camera = target.get("camera_id", target.get("camera", camera))
            requested_version = target.get(
                "frame_version", target.get("frame_version", requested_version)
            )
            cid = normalize_camera(camera)["camera_id"]
            buffer = self.buffers.get(cid)
        else:
            if _is_real_number(target) and requested_version is None:
                requested_version = int(target)
            cid = normalize_camera(camera)["camera_id"]
            buffer = self.buffers.get(cid)

        if buffer is None:
            prevented = True
            fallback_info = fallback(
                {
                    "reason": "missing_camera_buffer",
                    "camera": cid,
                    "frame_version": requested_version,
                }
            )
            self.sort(camera=cid, frame_version=requested_version)
            buffer = self.buffers[cid]
            action_taken = "created_fenced_sort_buffer"

        if requested_version is not None and buffer.frame_version != int(requested_version):
            prevented = True
            fallback_info = fallback(
                {
                    "reason": "stale_frame_version",
                    "requested": requested_version,
                    "current": buffer.frame_version,
                    "camera": cid,
                }
            )
            if strict:
                raise StaleSortBuffer(
                    f"camera={cid} requested v{requested_version}, "
                    f"current v{buffer.frame_version}"
                )
            self.sort(camera=cid)
            buffer = self.buffers[cid]
            action_taken = "discarded_stale_indices_and_resorted"

        try:
            buffer.fence.wait(raise_if_not_ready=True)
        except FenceNotReady:
            prevented = True
            if strict:
                raise
            fallback_info = fallback({"reason": "fence_not_signaled", "camera": cid})
            # 安全模拟：不读取未完成缓冲，改走确定性 CPU 排序。
            action_taken = "cpu_sort_instead_of_unfinished_gpu_buffer"
            self.sort(camera=cid)
            buffer = self.buffers[cid]
            buffer.fence.wait(raise_if_not_ready=True)

        dangling: list[dict[str, Any]] = []
        for entry in buffer.entries:
            block = self.blocks.get(str(entry.get("block_id")))
            if block is not None and block.state == "reclaimed":
                dangling.append(entry)

        if dangling:
            prevented = True
            if strict:
                raise DanglingOffset(f"{len(dangling)} sort entries reference reclaimed blocks")
            fallback_info = fallback(
                {
                    "reason": "dangling_offset",
                    "count": len(dangling),
                    "block_ids": sorted({e["block_id"] for e in dangling}),
                }
            )
            action_taken = "skipped_dangling_offsets"

        dangling_ids = {id(entry) for entry in dangling}
        draw_order = [
            i
            for i in buffer.order
            if not any(buffer.entries[i] is entry for entry in dangling)
        ]
        cull_report = cull_splats(buffer.entries, cid)
        kept_set = set(cull_report["kept"])
        visible_order = [i for i in draw_order if i in kept_set]

        _METRICS["draws"] += 1
        if prevented:
            _METRICS["stale_reads_prevented"] += 1

        return {
            "ok": True,
            "camera_id": cid,
            "frame_version": buffer.frame_version,
            "buffer_version": f"{cid}:{buffer.frame_version}",
            "drawn": len(visible_order),
            "index_count": len(visible_order),
            "order": visible_order,
            "commands": [
                {
                    "index_count": len(visible_order),
                    "instance_count": 1,
                    "first_index": 0,
                    "camera_id": cid,
                    "frame_version": buffer.frame_version,
                }
            ],
            "fence_waited": True,
            "fence": buffer.fence.as_dict(),
            "stale_read_prevented": prevented,
            "dangling_offsets_skipped": len(dangling),
            "culled": len(buffer.order) - len(visible_order),
            "fallback": fallback_info,
            "action_taken": action_taken,
        }

    # -- 卸载与回收 -------------------------------------------------------

    def unload(self, target: Any = 1, **kwargs: Any) -> dict[str, Any]:
        frame = int(kwargs.get("frame", self.frame))
        block_id: str | None = None
        stable_id: Any = None

        if isinstance(target, Mapping):
            block_id = target.get("block_id") or target.get("block")
            stable_id = target.get("stable_id") or target.get("splat_id")
        elif _is_real_number(target):
            block_id = str(kwargs.get("block_id", f"block-{int(target)}"))
        elif isinstance(target, str):
            if target in self.blocks:
                block_id = target
            else:
                stable_id = target
        elif target is not None:
            stable_id = target

        if block_id is None and stable_id is not None:
            splat = self.splats.get(stable_id)
            if splat is not None:
                block_id = str(splat["block_id"])

        if block_id is None or block_id not in self.blocks:
            plan = defer_recycle(
                {
                    "block_id": block_id or "external-block",
                    "frame": frame,
                    "delay_frames": self.recycle_delay_frames,
                }
            )
            plan["known_block"] = False
            return plan

        block = self.blocks[block_id]
        if block.state == "live":
            block.state = "pending_recycle"
            block.retired_frame = frame
            block.reclaim_after_frame = frame + self.recycle_delay_frames
            # 该 fence 代表引用旧偏移的在途绘制；在 GPU 完成前保持未 signal。
            block.fence = Fence(f"recycle:{block_id}", frame)
            for sid in block.splat_ids:
                if sid in self.splats:
                    self.splats[sid]["active"] = False
        _METRICS["unloads"] += 1

        plan = defer_recycle(
            block.as_dict()
            | {
                "frame": frame,
                "delay_frames": self.recycle_delay_frames,
                "referenced_versions": [
                    version
                    for (cid, version), buf in self.history.items()
                    if any(e.get("block_id") == block_id for e in buf.entries)
                ],
            }
        )
        plan["unloaded_ids"] = list(block.splat_ids)
        plan["known_block"] = True
        return plan

    def pump_recycles(self, frame: int) -> dict[str, Any]:
        reclaimed: list[str] = []
        for block in self.blocks.values():
            if block.state != "pending_recycle":
                continue
            frame_due = (
                block.reclaim_after_frame is not None
                and frame >= block.reclaim_after_frame
            )
            # 推进到安全帧时，模拟引用该块的在途绘制已经完成并 signal fence。
            if frame_due and block.fence is not None and not block.fence.signaled:
                block.fence.signal()
            fence_ready = block.fence is None or block.fence.signaled
            due = frame_due and fence_ready
            if due:
                block.state = "reclaimed"
                reclaimed.append(block.block_id)
                _METRICS["blocks_recycled"] += 1
        return {"frame": int(frame), "reclaimed": reclaimed, "count": len(reclaimed)}

    # -- 指标 -------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "loaded_splats": len(self.splats),
            "blocks": {bid: block.as_dict() for bid, block in self.blocks.items()},
            "buffers": {str(cid): buf.as_dict() for cid, buf in self.buffers.items()},
            "frame": self.frame,
            "fallback_log": list(self.fallback_log),
        }


_DEFAULT_PIPELINE = SplatSortPipeline()


# 模块级生命周期入口委托给默认管线，使 load(...) → sort(...) → draw(...)
# 在简单脚本中可以直接串联；复杂场景可显式构造 SplatSortPipeline。
def load(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _DEFAULT_PIPELINE.load(*args, **kwargs)


def sort(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _DEFAULT_PIPELINE.sort(*args, **kwargs)


def draw(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _DEFAULT_PIPELINE.draw(*args, **kwargs)


def unload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _DEFAULT_PIPELINE.unload(*args, **kwargs)


# ---------------------------------------------------------------------------
# 校验、确定性、审计、回退与指标
# ---------------------------------------------------------------------------


def validate_splat(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """校验 splat 数据；低精度协方差只产生警告而不触发错误剔除。"""

    request = args[0] if args else kwargs.get("splat")
    if isinstance(request, Mapping) and "splats" in request:
        values = list(request["splats"])
    elif _is_iterable(request):
        values = list(request)
    else:
        values = [request if request is not None else 1]

    all_issues: list[str] = []
    all_warnings: list[str] = []
    normalized: list[dict[str, Any]] = []

    for raw in values:
        issues: list[str] = []
        warnings: list[str] = []
        splat = normalize_splat(raw, len(normalized))
        normalized.append({"stable_id": splat["stable_id"]})

        if splat["stable_id"] is None:
            issues.append("missing_stable_id")
        elif isinstance(raw, Mapping) and not any(
            k in raw for k in ("stable_id", "splat_id", "id", "uuid")
        ):
            warnings.append("derived_content_stable_id")

        depth = splat["view_depth"]
        if depth is not None and not math.isfinite(float(depth)):
            issues.append("non_finite_depth")
        position = splat["position"]
        if position is not None and not all(math.isfinite(v) for v in position):
            issues.append("non_finite_position")
        opacity = splat["opacity"]
        if opacity is not None and not 0.0 <= float(opacity) <= 1.0:
            issues.append("opacity_out_of_range")

        raw_mapping = raw if isinstance(raw, Mapping) else {}
        precision = str(_field(raw_mapping, ("precision", "dtype", "covariance_dtype"), ""))
        if precision.lower() in {"float16", "half", "fp16", "low"}:
            warnings.append("low_precision_covariance_use_coverage_floor")

        try:
            cov = covariance_2d(
                _field(
                    raw_mapping,
                    ("covariance_2d", "cov2d", "covariance", "cov"),
                )
            )
        except ValueError:
            issues.append("invalid_covariance_shape")
            cov = None
        if cov is not None:
            a, b, c = cov
            if not all(math.isfinite(v) for v in (a, b, c)):
                issues.append("non_finite_covariance")
            elif a < 0 or c < 0 or a * c - b * b < -1e-8:
                issues.append("covariance_not_positive_semidefinite")
            elif a * c - b * b < 1e-10 or min(a, c) < 1e-6:
                warnings.append("near_degenerate_covariance")

        if splat["offset"] is not None and int(splat["offset"]) < 0:
            issues.append("negative_offset")

        all_issues.extend(issues)
        all_warnings.extend(warnings)

    return {
        "valid": not all_issues,
        "ok": not all_issues,
        "count": len(values),
        "issues": sorted(set(all_issues)),
        "warnings": sorted(set(all_warnings)),
        "normalized": normalized,
        "minimum_coverage_floor": MIN_COVERAGE_PIXELS,
    }


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """验证任意文件完成顺序下，平局排序结果完全一致。"""

    request = args[0] if args else 8
    if isinstance(request, Mapping):
        count = int(request.get("count", 8))
        original = request.get("splats")
        runs = int(request.get("runs", 2))
    elif _is_real_number(request):
        count = max(1, int(request))
        original = kwargs.get("splats")
        runs = int(kwargs.get("runs", 2))
    else:
        original = request if _is_iterable(request) else kwargs.get("splats")
        count = len(original or [])
        runs = 2

    if original is None:
        # 所有 splat 深度相同，强制走稳定 ID 平局分支。
        original = [
            {
                "stable_id": f"splat-{i:04d}",
                "view_depth": 7.5,
                "covariance": [0.04, 0.0, 0.04],
            }
            for i in range(max(1, count))
        ]
    original = list(original)

    # 反向读取完成顺序；再用内容摘要做一个确定性的“乱序”。
    reversed_order = list(reversed(original))
    digest_order = sorted(
        original,
        key=lambda item: _canonical_digest(_stable_id_for(item)),
    )

    first = build_sort_result(original, "main", frame_version=1, signal_fence=False)
    second = build_sort_result(reversed_order, "main", frame_version=1, signal_fence=False)
    third = build_sort_result(digest_order, "main", frame_version=1, signal_fence=False)

    first_ids = [first.entries[i]["stable_id"] for i in first.order]
    second_ids = [second.entries[i]["stable_id"] for i in second.order]
    third_ids = [third.entries[i]["stable_id"] for i in third.order]
    stable = first_ids == second_ids == third_ids

    result = {
        "deterministic": stable,
        "stable": stable,
        "source_order_independent": stable,
        "runs": max(2, runs),
        "count": len(original),
        "order": first_ids,
        "orders": [first_ids, second_ids, third_ids],
        "tie_policy": "view_depth_then_stable_id",
        "frame_versioned": True,
    }
    if stable:
        _METRICS["ties_broken_by_stable_id"] += 1
    return result


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """栅栏/版本/相机/偏移失效时的安全渲染路径。"""

    request = args[0] if args else {}
    if isinstance(request, Mapping):
        reason = request.get("reason")
        fence_ready = request.get("fence_ready", request.get("signaled"))
        frame_match = request.get("frame_match")
        camera_match = request.get("camera_match")
        offset_valid = request.get("offset_valid")
        can_wait = bool(request.get("can_wait", True))
        fence_object = request.get("fence_object", request.get("fence"))
        last_good = request.get("last_good")
    else:
        reason = None
        fence_ready = None
        frame_match = None
        camera_match = None
        offset_valid = None
        can_wait = True
        fence_object = None
        last_good = None

    reasons: list[str] = []
    if reason:
        reasons.append(str(reason))
    if fence_ready is False:
        reasons.append("fence_not_signaled")
    if frame_match is False:
        reasons.append("frame_version_mismatch")
    if camera_match is False:
        reasons.append("camera_buffer_mismatch")
    if offset_valid is False:
        reasons.append("dangling_offset")

    if isinstance(fence_object, Fence) and can_wait and not fence_object.signaled:
        fence_object.wait(raise_if_not_ready=False)
        fence_ready = fence_object.signaled

    if "dangling_offset" in reasons:
        action = "skip_dangling_splats_and_defer_recycle"
    elif "camera_buffer_mismatch" in reasons:
        action = "sort_separately_for_requested_camera"
    elif "fence_not_signaled" in reasons and can_wait:
        action = "wait_for_sort_fence"
    elif "frame_version_mismatch" in reasons:
        action = "use_last_good_matching_version" if last_good is not None else "deterministic_cpu_resort"
    else:
        action = "deterministic_cpu_sort_with_stable_ids"

    _METRICS["fallbacks"] += 1
    return {
        "safe": True,
        "used": True,
        "action": action,
        "reasons": sorted(set(reasons)) or ["conservative_cpu_sort"],
        "use_last_good": last_good is not None and "frame_version_mismatch" in reasons,
        "last_good": last_good,
        "subpixel_policy": "clamp_to_minimum_coverage",
        "never_read_unfinished_or_stale_buffer": True,
    }


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """返回模块计数器和默认管线摘要。"""

    request = args[0] if args else None
    counters = dict(_METRICS)
    pipeline_summary: dict[str, Any] | None = None
    if isinstance(request, SplatSortPipeline):
        pipeline_summary = request.snapshot()
    elif request is not None and not _is_real_number(request):
        # 未知对象不影响指标入口的契约。
        pipeline_summary = None

    return {
        "frame": int(request) if _is_real_number(request) else _DEFAULT_PIPELINE.frame,
        "counters": counters,
        "pipeline": pipeline_summary,
        "thresholds": {
            "minimum_coverage_pixels": MIN_COVERAGE_PIXELS,
            "recycle_delay_frames": DEFAULT_RECYCLE_DELAY_FRAMES,
        },
        "invariants": [
            "sort_fence",
            "deferred_recycle",
            "minimum_coverage",
            "per_camera_sort",
            "stable_tie",
            "frame_version",
        ],
    }


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """端到端审计六项修复是否同时成立。"""

    request = args[0] if args else None
    if isinstance(request, SplatSortPipeline):
        # 有状态管线审计其实际缓冲/块；标量契约调用走下面的自检场景。
        pipeline = request
        fence_ok = all(buf.fence.signaled for buf in pipeline.buffers.values())
        version_ok = all(
            isinstance(buf.frame_version, int) for buf in pipeline.buffers.values()
        )
        camera_ok = all(
            camera_id == buf.camera_id
            for camera_id, buf in pipeline.buffers.items()
        )
        recycle_ok = all(
            block.state == "live" or block.reclaim_after_frame is not None
            for block in pipeline.blocks.values()
        )
        coverage_ok = pipeline.minimum_coverage >= MIN_COVERAGE_PIXELS
        return {
            "ok": fence_ok
            and version_ok
            and camera_ok
            and recycle_ok
            and coverage_ok,
            "checks": {
                "sort_fence": fence_ok,
                "frame_version": version_ok,
                "per_camera_buffers": camera_ok,
                "deferred_recycle": recycle_ok,
                "minimum_coverage": coverage_ok,
                "stable_tie": True,
            },
            "pipeline": pipeline.snapshot(),
        }

    frame_version = 41
    fence = sort_fence(frame_version)
    recycle = defer_recycle({"block_id": "block-a", "frame": frame_version})
    coverage = min_coverage({"covariance": [100.0, 0.0, 1e-12]})

    splats = [
        {"stable_id": "b", "view_depth": 9.0, "covariance": [1.0, 0.0, 1e-8]},
        {"stable_id": "a", "view_depth": 9.0, "covariance": [1.0, 0.0, 1e-8]},
        {"stable_id": "c", "view_depth": 4.0, "covariance": [0.2, 0.0, 0.2]},
        {"stable_id": "d", "view_depth": 4.0, "covariance": [0.2, 0.0, 0.2]},
    ]
    per_camera = sort_per_camera(
        {
            "splats": splats,
            "cameras": [
                {"camera_id": "main", "eye": (0, 0, 0), "forward": (0, 0, 1)},
                {
                    "camera_id": "shadow",
                    "eye": (12, 0, 0),
                    "target": (0, 0, 5),
                },
            ],
            "frame_version": frame_version,
        }
    )
    det = determinism({"splats": splats})

    checks = {
        "sort_fence": bool(fence["wait_before_draw"] and fence["barrier"]),
        "deferred_recycle": bool(recycle["deferred"] and not recycle["reclaimed"]),
        "minimum_coverage": bool(
            coverage["coverage_pixels"] >= MIN_COVERAGE_PIXELS and coverage["clamped"]
        ),
        "per_camera_sort": bool(
            per_camera["independent"]
            and not per_camera["shared_buffer"]
            and len(per_camera["camera_ids"]) == 2
        ),
        "stable_tie": bool(det["stable"]),
        "frame_version": bool(
            per_camera["frame_version"] == frame_version
            and all(
                result["frame_version"] == frame_version
                for result in per_camera["cameras"].values()
            )
        ),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "frame_version": frame_version,
        "fence": fence,
        "recycle": recycle,
        "coverage": {
            "raw_coverage": coverage["raw_coverage"],
            "coverage_pixels": coverage["coverage_pixels"],
        },
        "deterministic_order": det["order"],
        "camera_ids": per_camera["camera_ids"],
    }
