"""XuanShu H3 太极装甲 v3s1 — v3s 优化版节点（四类优化）。

v3s1 = v3s + 四类优化：
a) 判定计数改读 args["timestep"] 按真实去噪步对齐：用 timestep 变化驱动 step_idx，
   同一去噪步内的多次 UNet 调用（如 CFG 正负条件）不再被盲计为多步；
   timestep 缺失/不可解析时回退 v3s 原调用计数逻辑（计算步末尾 step_idx += 1）。
b) prev_output_shape 输出形状保护：缓存输出与当前输入 batch 结构不匹配时
   自动 reset 状态并强制计算，避免复用形状不兼容的旧缓存输出。
c) 整批缓存分块：batch>1 时按 chunk_size 分块计算 delta 并取各块最大值（最保守），
   避免整批平均掩盖局部漂移导致误缓存。实现选择：整批分块。
d) 自定义采样器旁路：非标准采样路径（args 缺 input/timestep/c 协议键）直接
   透传 apply_model，不参与缓存决策与状态更新。

v3s 原有全部参数、默认值与保护逻辑（IEC、形状变化保护、多项式校准、误差熔断等）不变。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np
import torch


@dataclasses.dataclass
class TaijiArmorV3S1State:
    step_idx: int = 0
    last_ts: Optional[float] = None
    reported: bool = False
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    prev_output_shape: Optional[tuple] = None
    prev_prev_input: Optional[torch.Tensor] = None

    yang_steps: list = dataclasses.field(default_factory=list)
    yin_steps: list = dataclasses.field(default_factory=list)
    reuse_count: int = 0
    real_count: int = 0

    strategy_scores: dict = dataclasses.field(default_factory=dict)
    current_strategy: str = "l1"

    anomaly_count: int = 0
    fallback_active: bool = False

    peak_vram_mb: int = 0

    # v3 原有
    delta_history: list = dataclasses.field(default_factory=list)
    poly_coeffs: Optional[list] = None
    accumulated_error: float = 0.0
    consecutive_cache: int = 0
    max_consecutive_hit: int = 0
    output_trend: Optional[torch.Tensor] = None
    correction_count: int = 0
    confidence_history: list = dataclasses.field(default_factory=list)
    poly_thresh_used: int = 0
    strat_thresh_used: int = 0

    # wall-clock timing
    step_wall_cumulative: float = 0.0
    step_wall_times: list = dataclasses.field(default_factory=list)

    # v3s 新增：IEC
    iec_error_buffer: list = dataclasses.field(default_factory=list)
    iec_max_buffer: int = 16
    iec_penalty_log: list = dataclasses.field(default_factory=list)
    prev_input_shape: Optional[tuple] = None

    def reset(self, keep_timeline: bool = False) -> None:
        """重置全部缓存状态。keep_timeline=True 时保留 step_idx/last_ts/reported
        （真实去噪步时间线不被形状保护重置打断）。"""
        step_idx_keep = self.step_idx
        last_ts_keep = self.last_ts
        reported_keep = self.reported
        self.step_idx = 0
        self.last_ts = None
        self.reported = False
        self.prev_input = None
        self.prev_output = None
        self.prev_output_shape = None
        self.prev_prev_input = None
        self.yang_steps = []
        self.yin_steps = []
        self.reuse_count = 0
        self.real_count = 0
        self.strategy_scores = {}
        self.current_strategy = "l1"
        self.anomaly_count = 0
        self.fallback_active = False
        self.peak_vram_mb = 0
        self.delta_history = []
        self.poly_coeffs = None
        self.accumulated_error = 0.0
        self.consecutive_cache = 0
        self.max_consecutive_hit = 0
        self.output_trend = None
        self.correction_count = 0
        self.confidence_history = []
        self.poly_thresh_used = 0
        self.strat_thresh_used = 0
        self.step_wall_cumulative = 0.0
        self.step_wall_times = []
        self.iec_error_buffer = []
        self.iec_penalty_log = []
        self.prev_input_shape = None
        if keep_timeline:
            self.step_idx = step_idx_keep
            self.last_ts = last_ts_keep
            self.reported = reported_keep

    def accumulate_strategy(self, strategy: str, delta: float) -> None:
        d = self.strategy_scores.setdefault(strategy, {"mean": 0.0, "m2": 0.0, "n": 0})
        n = d["n"]
        new_n = n + 1
        old_mean = d["mean"]
        d["mean"] = old_mean + (delta - old_mean) / new_n
        d["m2"] = d["m2"] + (delta - old_mean) * (delta - d["mean"])
        d["n"] = new_n

    def strategy_variance(self, strategy: str) -> float:
        d = self.strategy_scores.get(strategy, {})
        n = d.get("n", 0)
        if n < 2:
            return 0.0
        return d["m2"] / (n - 1)


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def timestep_residual(current: torch.Tensor, previous: torch.Tensor,
                      prev_prev: Optional[torch.Tensor] = None) -> float:
    if prev_prev is None:
        return rel_l1(current, previous)
    d1 = (previous - prev_prev).abs().mean()
    d2 = (current.to(previous.dtype) - previous).abs().mean()
    return (d2 - d1).abs().clamp(min=1e-8).item()


def attn_var_signal(current: torch.Tensor) -> float:
    return current.var().item()


def read_vram_mb() -> int:
    try:
        return int(torch.cuda.memory_reserved() // (1024 * 1024))
    except Exception:
        return 0


class TaijiSchedulerV3S1:
    STRATEGIES = {
        "l1":        {"thresh": 0.15, "signal": "l1"},
        "conserv":   {"thresh": 0.10, "signal": "l1"},
        "aggr":      {"thresh": 0.20, "signal": "l1"},
        "residual":  {"thresh": 0.15, "signal": "residual"},
        "weighted":  {"thresh": 0.15, "signal": "weighted"},
    }

    def __init__(self, total_steps: int, warmup_steps: int, yang_thresh: float,
                 adaptive: bool, enable_ensemble: bool, vram_thresh_mb: int,
                 fallback_mode: bool, end_step: int, verbose: bool,
                 adaptive_scale: float = 2.5,
                 poly_calibrate: bool = True,
                 poly_degree: int = 2,
                 safety_margin: float = 1.3,
                 error_budget: float = 0.5,
                 max_consecutive: int = 4,
                 correction_strength: float = 0.3,
                 confidence_mode: bool = True,
                 # v3s 新增
                 iec_enabled: bool = True,
                 iec_gamma: float = 0.6,
                 # v3s1 新增
                 chunk_size: int = 2):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.yang_thresh = yang_thresh
        self.adaptive = adaptive
        self.enable_ensemble = enable_ensemble
        self.vram_thresh_mb = vram_thresh_mb
        self.fallback_mode = fallback_mode
        self.end_step = end_step
        self.verbose = verbose
        self.adaptive_scale = adaptive_scale

        self.poly_calibrate = poly_calibrate
        self.poly_degree = poly_degree
        self.safety_margin = safety_margin
        self.error_budget = error_budget
        self.max_consecutive = max_consecutive
        self.correction_strength = correction_strength
        self.confidence_mode = confidence_mode

        self.iec_enabled = iec_enabled
        self.iec_gamma = iec_gamma

        self.chunk_size = max(1, int(chunk_size))

    def _vram_gate(self, state: TaijiArmorV3S1State) -> bool:
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        pressure = vram > self.vram_thresh_mb
        if self.verbose and pressure:
            print("[TaijiArmor v3s1] VRAM水位: %dMB > %dMB → 阈值×1.5" % (vram, self.vram_thresh_mb))
        return pressure

    def _compute_signal(self, strategy: str, cur: torch.Tensor,
                        prev: torch.Tensor, prev_prev: Optional[torch.Tensor]) -> float:
        if strategy == "residual":
            return timestep_residual(cur, prev, prev_prev)
        if strategy == "weighted":
            l1 = rel_l1(cur, prev)
            res = timestep_residual(cur, prev, prev_prev)
            var = attn_var_signal(cur) * 0.01
            return 0.6 * l1 + 0.3 * res + 0.1 * var
        return rel_l1(cur, prev)

    def _compute_signal_chunked(self, strategy: str, cur: torch.Tensor,
                                prev: torch.Tensor,
                                prev_prev: Optional[torch.Tensor]) -> float:
        """v3s1 (c) 整批缓存分块：batch>1 时按 chunk_size 分块计算 delta，
        取各块最大值（最保守），避免整批平均掩盖局部漂移导致误缓存。"""
        b = cur.shape[0]
        if b <= 1 or b <= self.chunk_size:
            return self._compute_signal(strategy, cur, prev, prev_prev)
        deltas = []
        for i in range(0, b, self.chunk_size):
            c = cur[i:i + self.chunk_size]
            p = prev[i:i + self.chunk_size]
            pp = prev_prev[i:i + self.chunk_size] if prev_prev is not None else None
            deltas.append(self._compute_signal(strategy, c, p, pp))
        return max(deltas)

    def _fitness(self, strategy: str, state: TaijiArmorV3S1State) -> float:
        d = state.strategy_scores.get(strategy, {})
        mean = d.get("mean", 0.0)
        var = state.strategy_variance(strategy)
        return 1.0 / (1.0 + mean * 5.0 + var * 10.0)

    def _select_strategy(self, state: TaijiArmorV3S1State) -> str:
        best = "l1"
        best_fit = -1.0
        for s in self.STRATEGIES:
            fit = self._fitness(s, state)
            if fit > best_fit:
                best_fit = fit
                best = s
        state.current_strategy = best
        return best

    def _fallback_threshold(self) -> float:
        effective_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        cacheable = max(0, effective_end - self.warmup_steps)
        max_reuse_ratio = cacheable / max(1, self.total_steps)
        return min(0.9, max_reuse_ratio * 0.85)

    def _fit_polynomial(self, state: TaijiArmorV3S1State) -> None:
        if len(state.delta_history) < 2:
            return
        steps = np.array([d[0] for d in state.delta_history], dtype=np.float64)
        deltas = np.array([d[1] for d in state.delta_history], dtype=np.float64)
        log_deltas = np.log(np.maximum(deltas, 1e-8))
        degree = min(self.poly_degree, len(steps) - 1)
        try:
            coeffs = np.polyfit(steps, log_deltas, degree)
            state.poly_coeffs = coeffs.tolist()
            if self.verbose:
                predicted_strs = []
                for s in range(int(steps.min()), int(steps.max()) + 4):
                    pred = float(np.exp(np.polyval(coeffs, s)))
                    predicted_strs.append("s%d→%.4f" % (s, pred))
                print("[TaijiArmor v3s1] 对数线性拟合(deg=%d, n=%d): %s" % (
                    degree, len(steps), " ".join(predicted_strs)))
        except Exception as e:
            if self.verbose:
                print("[TaijiArmor v3s1] 对数线性拟合失败: %s" % e)

    def _predict_delta(self, step: int, state: TaijiArmorV3S1State) -> Optional[float]:
        if state.poly_coeffs is None:
            return None
        log_val = float(np.polyval(state.poly_coeffs, step))
        return max(float(np.exp(log_val)), 0.001)

    def _compute_confidence(self, delta: float, thresh: float) -> float:
        if not self.confidence_mode:
            return 1.0 if delta < thresh else 0.0
        if delta < thresh * 0.5:
            return 1.0
        elif delta < thresh:
            return (thresh - delta) / (thresh * 0.5)
        return 0.0

    def _compute_iec_penalty(self, state: TaijiArmorV3S1State) -> float:
        """IEC：从历史误差缓冲计算指数加权惩罚值。"""
        if not self.iec_enabled or len(state.iec_error_buffer) < 3:
            return 0.0
        errors = torch.tensor([e for _, e in state.iec_error_buffer],
                              dtype=torch.float32)
        weights = torch.exp(torch.linspace(-2.0, 0.0, len(errors)))
        weights = weights / weights.sum()
        corrected_error = (errors * weights).sum().item()
        return corrected_error * self.iec_gamma

    def should_reuse(self, state: TaijiArmorV3S1State, current_input: torch.Tensor) -> tuple:
        step = state.step_idx

        # === 增强形状变化保护：入口最前方检测（v3s 原有） ===
        cur_shape = tuple(current_input.shape)
        if state.prev_input_shape is not None and cur_shape != state.prev_input_shape:
            if self.verbose:
                print("[TaijiArmor v3s1] 形状变化 %s → %s → 自动重置 state" % (
                    state.prev_input_shape, cur_shape))
            state.reset()
            state.prev_input_shape = cur_shape
            return (False, 0.0, 0.0, 0.0, "shape_reset")

        if state.prev_output is None or state.prev_input is None:
            state.prev_input_shape = cur_shape
            return (False, 0.0, 0.0, 0.0, "no_ref")

        # 始终计算 delta（v3s1 (c)：整批分块）
        strategy = self.current_strategy if not self.enable_ensemble else self._select_strategy(state)
        prev_input = state.prev_input
        prev_prev = state.prev_prev_input
        cur_cpu = current_input.detach().cpu()
        delta = self._compute_signal_chunked(strategy, cur_cpu, prev_input, prev_prev)
        state.accumulate_strategy(strategy, delta)
        state.delta_history.append((step, delta, strategy))

        vram_pressure = self._vram_gate(state)

        # warmup
        if step < self.warmup_steps:
            state.yang_steps.append(step)
            if self.poly_calibrate:
                self._fit_polynomial(state)
            return (False, delta, 0.0, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        effective_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        if step >= effective_end:
            state.yang_steps.append(step)
            return (False, delta, 0.0, 0.0, "after_end(%d>=%d)" % (step, effective_end))

        if self.fallback_mode and state.fallback_active:
            state.yang_steps.append(step)
            return (False, delta, 0.0, 0.0, "fallback")

        # === 阈值计算：多项式优先，回退策略 ===
        thresh = None
        thresh_source = None

        if self.poly_calibrate and state.poly_coeffs is not None:
            predicted = self._predict_delta(step, state)
            if predicted is not None and predicted > 0:
                thresh = predicted * self.safety_margin
                thresh_source = "poly"
                state.poly_thresh_used += 1

        if thresh is None:
            thresh = self.STRATEGIES[strategy]["thresh"]
            thresh_source = "strat"
            state.strat_thresh_used += 1
            if self.adaptive:
                progress = step / max(1, self.total_steps)
                if progress >= 0.3:
                    thresh *= self.adaptive_scale

        if vram_pressure:
            thresh *= 1.5

        if thresh_source == "strat":
            thresh = max(thresh, self.yang_thresh)
        else:
            thresh = max(thresh, 0.01)

        # === IEC 阈值惩罚 ===
        iec_penalty = self._compute_iec_penalty(state)
        if iec_penalty > 0:
            original_thresh = thresh
            thresh = max(thresh - iec_penalty, thresh * 0.2)
            state.iec_penalty_log.append((step, iec_penalty, original_thresh, thresh))
            if self.verbose:
                print("[TaijiArmor v3s1] IEC惩罚: %.4f → 阈值 %.4f→%.4f" % (
                    iec_penalty, original_thresh, thresh))

        # === 误差累积熔断 ===
        if state.accumulated_error + delta > self.error_budget and state.consecutive_cache > 0:
            if self.verbose:
                print("[TaijiArmor v3s1] 误差熔断: 累积=%.3f + 当前=%.3f > 预算=%.3f → 强制计算" % (
                    state.accumulated_error, delta, self.error_budget))
            state.accumulated_error = 0.0
            state.consecutive_cache = 0
            state.yang_steps.append(step)
            return (False, delta, thresh, 0.0, "error_break(%.2f)" % (state.accumulated_error + delta))

        # === 连续缓存上限 ===
        if state.consecutive_cache >= self.max_consecutive:
            if self.verbose:
                print("[TaijiArmor v3s1] 连续缓存上限: %d >= %d → 强制计算" % (
                    state.consecutive_cache, self.max_consecutive))
            state.consecutive_cache = 0
            state.accumulated_error = 0.0
            state.yang_steps.append(step)
            return (False, delta, thresh, 0.0, "max_consec(%d)" % self.max_consecutive)

        # === 置信度计算 ===
        confidence = self._compute_confidence(delta, thresh)
        state.confidence_history.append((step, confidence))

        # === 缓存决策 ===
        if confidence > 0.5:
            state.yin_steps.append(step)
            state.accumulated_error += delta
            state.consecutive_cache += 1
            if state.consecutive_cache > state.max_consecutive_hit:
                state.max_consecutive_hit = state.consecutive_cache
            return (True, delta, thresh, confidence, "%s(%.2f)" % (strategy, confidence))

        state.yang_steps.append(step)
        state.consecutive_cache = 0
        return (False, delta, thresh, confidence, "%s(%.2f)" % (strategy, confidence))


def _is_standard_sampling_path(args) -> bool:
    """v3s1 (d) 自定义采样器旁路判定：标准采样路径必须带 input/timestep/c 协议键。"""
    return isinstance(args, dict) and "input" in args and "timestep" in args and "c" in args


def make_armor_wrapper_v3s1(
    state: TaijiArmorV3S1State,
    total_steps: int,
    warmup_steps: int,
    yang_thresh: float,
    adaptive: bool,
    enable_ensemble: bool,
    vram_thresh_mb: int,
    fallback_mode: bool,
    end_step: int,
    verbose: bool,
    adaptive_scale: float = 2.5,
    poly_calibrate: bool = True,
    poly_degree: int = 2,
    safety_margin: float = 1.3,
    error_budget: float = 0.5,
    max_consecutive: int = 4,
    correction_strength: float = 0.3,
    confidence_mode: bool = True,
    iec_enabled: bool = True,
    iec_gamma: float = 0.6,
    chunk_size: int = 2,
) -> Callable[..., torch.Tensor]:
    scheduler = TaijiSchedulerV3S1(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        yang_thresh=yang_thresh,
        adaptive=adaptive,
        enable_ensemble=enable_ensemble,
        vram_thresh_mb=vram_thresh_mb,
        fallback_mode=fallback_mode,
        end_step=end_step,
        verbose=verbose,
        adaptive_scale=adaptive_scale,
        poly_calibrate=poly_calibrate,
        poly_degree=poly_degree,
        safety_margin=safety_margin,
        error_budget=error_budget,
        max_consecutive=max_consecutive,
        correction_strength=correction_strength,
        confidence_mode=confidence_mode,
        iec_enabled=iec_enabled,
        iec_gamma=iec_gamma,
        chunk_size=chunk_size,
    )

    import time as _time
    _step_t0 = [_time.perf_counter()]

    def _parse_timestep(ts_raw):
        """v3s1 (a) 从 args["timestep"] 提取标量去噪步；不可解析返回 None。"""
        if ts_raw is None:
            return None
        try:
            if hasattr(ts_raw, "detach"):
                return float(ts_raw.detach().cpu().float().reshape(-1).max())
            return float(ts_raw)
        except Exception:
            return None

    def _maybe_report(state_inner, ts_mode_inner):
        if state_inner.step_idx >= max(0, total_steps - 1) and not state_inner.reported:
            state_inner.reported = True
            _report_v3s1(state_inner, total_steps, warmup_steps, yang_thresh,
                         adaptive, enable_ensemble, vram_thresh_mb,
                         fallback_mode, end_step, verbose,
                         poly_calibrate, safety_margin, error_budget,
                         max_consecutive, correction_strength, confidence_mode,
                         iec_enabled, iec_gamma, chunk_size)

    def wrapper(apply_model, args) -> torch.Tensor:
        # === v3s1 (d) 自定义采样器旁路：非标准路径直接透传不缓存 ===
        if not _is_standard_sampling_path(args):
            if verbose:
                print("[TaijiArmor v3s1] 非标准采样路径，旁路透传（不缓存）")
            x = args["input"]
            return apply_model(x, args.get("timestep"), **args.get("c", {}))

        x = args["input"]
        _t0 = _time.perf_counter()
        if _step_t0[0] is None:
            _step_t0[0] = _t0

        # === v3s1 (a) 真实去噪步对齐 ===
        ts_val = _parse_timestep(args.get("timestep"))
        ts_mode = ts_val is not None
        if ts_mode:
            if state.last_ts is not None and abs(ts_val - state.last_ts) > 1e-3:
                state.step_idx += 1
            state.last_ts = ts_val
        # ts_mode=False：回退 v3s 原调用计数逻辑（计算步末尾 step_idx += 1）

        # === v3s1 (b) prev_output_shape 输出形状保护 ===
        if state.prev_output_shape is not None and state.prev_output_shape[0] != x.shape[0]:
            if verbose:
                print("[TaijiArmor v3s1] 输出形状保护：缓存输出 batch=%d ≠ 当前输入 batch=%d → reset 并强制计算" % (
                    state.prev_output_shape[0], x.shape[0]))
            state.reset(keep_timeline=True)

        reuse, delta, thresh, confidence, strategy = scheduler.should_reuse(state, x)
        _sched_dt = _time.perf_counter() - _t0
        _step_wall = _time.perf_counter() - _step_t0[0]

        if reuse:
            state.reuse_count += 1
            state.step_wall_times.append(_step_wall)
            state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall

            if correction_strength > 0 and state.output_trend is not None:
                trend = state.output_trend
                corrected = state.prev_output + trend * correction_strength
                state.correction_count += 1
                if verbose:
                    print("[TaijiArmor v3s1] Step %d/%d: CACHE+修正 wall=%.1fs (delta=%.4f thresh=%.4f conf=%.2f strat=%s trend_norm=%.4f)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh, confidence, strategy,
                        trend.abs().mean().item()))
                _maybe_report(state, ts_mode)
                return corrected.to(x.device)
            else:
                if verbose:
                    print("[TaijiArmor v3s1] Step %d/%d: CACHE wall=%.1fs (delta=%.4f thresh=%.4f conf=%.2f strat=%s)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh, confidence, strategy))
                _maybe_report(state, ts_mode)
                return state.prev_output.to(x.device)

        # 计算步
        if fallback_mode:
            total_done = state.reuse_count + state.real_count
            fb_thresh = scheduler._fallback_threshold()
            if total_done > 3 and state.reuse_count / total_done > fb_thresh:
                state.anomaly_count += 1
                if state.anomaly_count >= 2:
                    state.fallback_active = True
                    if verbose:
                        print("[TaijiArmor v3s1] 护体真元触发：命中率 "
                              "%d/%d > %.2f → 回退全算" % (
                              state.reuse_count, total_done, fb_thresh))
            else:
                state.anomaly_count = 0

        output = apply_model(x, args["timestep"], **args["c"])

        # === 更新趋势（EMA of output changes） ===
        if state.prev_output is not None and correction_strength > 0:
            prev_out = state.prev_output
            cur_out_cpu = output.detach().cpu()
            change = cur_out_cpu - prev_out
            if state.output_trend is None:
                state.output_trend = change
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change

        # === IEC 误差记录 ===
        if iec_enabled and delta > 0 and state.step_idx > 0:
            state.iec_error_buffer.append((state.step_idx, delta))
            if len(state.iec_error_buffer) > state.iec_max_buffer:
                state.iec_error_buffer.pop(0)

        state.prev_prev_input = state.prev_input if state.prev_input is not None else None
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.prev_output_shape = tuple(output.shape)
        state.prev_input_shape = tuple(x.shape)
        state.real_count += 1
        state.step_wall_times.append(_step_wall)
        state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall
        _model_dt = _time.perf_counter() - _t0
        if verbose:
            print("[TaijiArmor v3s1] Step %d/%d: COMPUTE wall=%.1fs (sched=%.0fms model=%.0fms) (%s delta=%.4f thresh=%.4f conf=%.2f)" % (
                state.step_idx, total_steps, _step_wall, _sched_dt*1000, _model_dt*1000,
                strategy, delta, thresh, confidence))
        if not ts_mode:
            # 回退 v3s 原逻辑：计算步末尾计数
            state.step_idx += 1
        _step_t0[0] = _time.perf_counter()
        _maybe_report(state, ts_mode)
        return output

    return wrapper


def _report_v3s1(state, total_steps, warmup_steps, yang_thresh, adaptive,
                 enable_ensemble, vram_thresh_mb, fallback_mode, end_step, verbose,
                 poly_calibrate, safety_margin, error_budget,
                 max_consecutive, correction_strength, confidence_mode,
                 iec_enabled, iec_gamma, chunk_size):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    wall_total = state.step_wall_cumulative
    wall_times_str = ""
    if state.step_wall_times:
        wall_times_str = " | wall_times=[" + ", ".join("%.1f" % t for t in state.step_wall_times) + "]"
    iec_str = ""
    if iec_enabled and state.iec_penalty_log:
        avg_pen = sum(p[1] for p in state.iec_penalty_log) / len(state.iec_penalty_log)
        iec_str = f"\n  v3s1 IEC: enabled={iec_enabled} gamma={iec_gamma} " \
                  f"buffer_size={len(state.iec_error_buffer)} " \
                  f"penalty_count={len(state.iec_penalty_log)} avg_penalty={avg_pen:.4f}"
    base = (f"[XuanShuTaijiArmor v3s1] steps={total_steps} warmup={warmup_steps} "
            f"yang_thresh={yang_thresh} adaptive={adaptive} ensemble={enable_ensemble} "
            f"vram_th={vram_thresh_mb}MB fallback={fallback_mode} end_step={end_step} "
            f"chunk_size={chunk_size} "
            f"yang={len(state.yang_steps)} yin={len(state.yin_steps)} "
            f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}% "
            f"peak_vram={state.peak_vram_mb}MB | total_wall={wall_total:.1f}s\n"
            f"  v3: poly={poly_calibrate} safety={safety_margin} err_budget={error_budget} "
            f"max_consec={max_consecutive} correction={correction_strength} conf_mode={confidence_mode}\n"
            f"  v3: poly_thresh_used={state.poly_thresh_used} strat_thresh_used={state.strat_thresh_used} "
            f"max_consec_hit={state.max_consecutive_hit} corrections={state.correction_count} "
            f"final_accumulated_err={state.accumulated_error:.4f}{wall_times_str}{iec_str}")
    if verbose:
        base += (f"\n  yang_steps={state.yang_steps}\n  yin_steps={state.yin_steps}\n"
                 f"  strategy={state.current_strategy}\n"
                 f"  delta_history={[(d[0], round(d[1],4), d[2]) for d in state.delta_history]}\n"
                 f"  confidence={[(c[0], round(c[1],2)) for c in state.confidence_history]}")
        if state.iec_penalty_log:
            base += f"\n  iec_penalties={[(p[0], round(p[1],4), round(p[2],4), round(p[3],4)) for p in state.iec_penalty_log]}"
    print(base)


class MiniMaxH3TaijiArmorV3S1:
    """玄枢 v3s1：v3s 优化版节点。

    a) timestep 真实步对齐；b) prev_output_shape 输出形状保护；
    c) 整批缓存分块（chunk_size 分块取保守 delta）；d) 自定义采样器旁路透传。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                           "tooltip": "阳·刚步保护：前 N 步全算"}),
                "yang_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "最低阈值下限：实际阈值不低于此值"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护：最后 N 步全算（保护细节）"}),
                "adaptive": ("BOOLEAN", {"default": True,
                                            "tooltip": "自适应收敛：前期探索、后期保守"}),
                "enable_ensemble": ("BOOLEAN", {"default": True,
                                                    "tooltip": "养蛊择优：多策略选最稳"}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                                              "tooltip": "显存水位阈值(MB)：超限自动更保守"}),
                "fallback_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "护体真元：异常自动回退全算"}),
                "verbose": ("BOOLEAN", {"default": True}),
                "adaptive_scale": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                                               "tooltip": "v2自适应倍率：progress>=0.3时 thresh × 此值"}),
                "poly_calibrate": ("BOOLEAN", {"default": True,
                                                "tooltip": "v3多项式校准：warmup期delta拟合多项式预测逐步阈值"}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5,
                                          "tooltip": "多项式阶数：2=二次（推荐），1=线性"}),
                "safety_margin": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1,
                                              "tooltip": "安全余量：thresh = 预测delta × 此值"}),
                "error_budget": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05,
                                             "tooltip": "误差预算：连续缓存累积误差超此值时强制计算"}),
                "max_consecutive": ("INT", {"default": 4, "min": 1, "max": 10,
                                              "tooltip": "连续缓存上限：连续命中超此值时强制计算"}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "缓存补偿强度：趋势外推修正因子"}),
                "confidence_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "v3置信度衰减：用连续概率替代二值缓存决策"}),
                # === v3s 新增 ===
                "iec_enabled": ("BOOLEAN", {"default": True,
                                             "tooltip": "IEC迭代误差校正：历史误差指数加权 → 阈值惩罚，动态收紧缓存"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1,
                                         "tooltip": "IEC衰减因子：越大惩罚越强(0.6=适中)"}),
                # === v3s1 新增 ===
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 8,
                                         "tooltip": "v3s1 整批分块：batch>1 时按此块大小分块计算 delta 取最大（保守）"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v3s1（玄枢 v3s1）"

    def apply(self, model, total_steps, warmup_steps, yang_thresh, end_step,
              adaptive, enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              adaptive_scale,
              poly_calibrate, poly_degree, safety_margin, error_budget,
              max_consecutive, correction_strength, confidence_mode,
              iec_enabled, iec_gamma,
              chunk_size):
        if warmup_steps >= total_steps:
            print(f"[TaijiArmor v3s1] 警告：warmup({warmup_steps}) >= total({total_steps})，缓存完全失效")

        if "xuanshu_teacache_guard" in model.model_options:
            print("[TaijiArmor v3s1] 警告：检测到 Guard 已应用，装甲 wrapper 将被覆盖")

        state = TaijiArmorV3S1State()
        state.reset()
        wrapper = make_armor_wrapper_v3s1(
            state=state,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            yang_thresh=yang_thresh,
            adaptive=adaptive,
            enable_ensemble=enable_ensemble,
            vram_thresh_mb=vram_thresh_mb,
            fallback_mode=fallback_mode,
            end_step=end_step,
            verbose=verbose,
            adaptive_scale=adaptive_scale,
            poly_calibrate=poly_calibrate,
            poly_degree=poly_degree,
            safety_margin=safety_margin,
            error_budget=error_budget,
            max_consecutive=max_consecutive,
            correction_strength=correction_strength,
            confidence_mode=confidence_mode,
            iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
            chunk_size=chunk_size,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v3s1"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        return (patched,)


NODE_CLASS_MAPPINGS_V3S1 = {"MiniMaxH3TaijiArmorV3S1": MiniMaxH3TaijiArmorV3S1}
NODE_DISPLAY_NAME_MAPPINGS_V3S1 = {"MiniMaxH3TaijiArmorV3S1": "末梦秋花v3s1(顺风飘)"}
