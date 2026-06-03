import os
import json
import time
import re
import datetime
import subprocess
import threading
import queue
from openai import OpenAI

# ── 固定配置（不是 agent 决策，是写死的实验设置）──────────────
LR_INIT        = 3e-4        # 初始 LR（固定）
LR_FLOOR       = 1e-6        # LR 下限
LR_CEIL        = 5e-2        # LR 上限
TOTAL_STEPS    = 10000       # 1.2B tokens @ 512×256 tok/step
WARMUP_STEPS   = 1000        # warmup 固定 1000 步
DECISION_EVERY = int(os.environ.get("K", "300"))  # K 固定 300（≈API延迟下的实际下限；可设 K=100/500 做 ablation）
SAVE_EVERY     = DECISION_EVERY                    # ckpt 与决策点对齐
# SafetyGuard（纯规则，无 LLM）：Controller 提议的 LR 被钳制在 [0.5x, 2x] 当前值
LR_JUMP_LOW    = 0.5
LR_JUMP_HIGH   = 2.0
MAX_FAST_CRASH = 20          # 安全护栏：连续快崩 20 次就放弃当前 run
# ── 紧急 self-healing 触发器（不调 Qwen，纯规则） ──
EMERGENCY_LOSS_JUMP = 2.0    # loss 单次跳涨 > 此值 → 紧急降 LR
EMERGENCY_LOSS_MAX  = 20.0   # loss 绝对值 > 此值 → 紧急降 LR
EMERGENCY_LR_FACTOR = 0.3    # 紧急情况：LR 砍到 30%
# ── 智能回退（崩溃后选 ckpt + 自动降 LR）──
ROLLBACK_LR_FACTOR  = 0.5    # 每次 rollback 自动 LR 砍半（防止再炸）
ROLLBACK_BACK_STEPS = 1000   # 回退到崩溃步前至少 1000 步的最近 ckpt
# ── 可通过环境变量切换模型规模 / 隔离经验回放 ──
# MODEL_CONFIG: 训练用的模型 config（默认 60M）；小模型代理搜 schedule 时换成 llama_20m/10m
MODEL_CONFIG   = os.environ.get("MODEL_CONFIG", "cola_configs/llama_60m.json")
# RUN_TAG: 给 save_dir + 三个经验回放 jsonl 加后缀，避免不同模型规模的历史互相污染
#          例如 RUN_TAG=llama20m → results/agent_run_llama20m + decisions_log_llama20m.jsonl
RUN_TAG        = os.environ.get("RUN_TAG", "").strip()
_SFX           = f"_{RUN_TAG}" if RUN_TAG else ""
# wandb project：所有后续 2-agent 实验落在 2AgentPlan 项目下
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "2AgentPlan")

# ════════════════════════════════════════════════════════════════════════
#  ★ 配置页：2 个 Agent 各用哪个模型当"大脑"（SafetyGuard 是纯规则，无模型）★
#  - 走 DashScope（OpenAI 兼容）client；填你账户支持的 model id。
#  - 两档：STRONG=重推理/低频，FAST=低延迟/高频。默认都用已验证的 qwen3.6-plus。
# ════════════════════════════════════════════════════════════════════════
_MODEL_STRONG = os.environ.get("MODEL_STRONG", "qwen3.6-plus-2026-04-02")  # 强推理档
_MODEL_FAST   = os.environ.get("MODEL_FAST",   "qwen3.6-plus-2026-04-02")  # 快/省档
AGENT_MODELS = {
    "controller": _MODEL_FAST,    # [Agent] 每 K 步调 LR（基于当前 loss + Memory 检索）—— 高频
    "memory":     _MODEL_STRONG,  # [Agent] 跨回合策略检索器 —— 低频、重综合分析
    # SafetyGuard = 纯代码规则（崩溃回滚 + LR 跳变约束 [0.5x,2x]），不用模型
}

STATE_FILE     = "training_state.json"   # 实时通信文件，全局（同时只跑一个 run）
LR_CMD_FILE    = "lr_command.txt"
QWEN_LOG_FILE  = f"agent_qwen{_SFX}.log"
ROLLBACK_LOG_FILE = f"rollback_log{_SFX}.jsonl"   # 持久化每次 rollback 事件
DECISIONS_LOG_FILE = f"decisions_log{_SFX}.jsonl" # 持久化每次 Qwen 决策
FINAL_RESULTS_LOG_FILE = f"final_results_log{_SFX}.jsonl"   # 持久化每次 run 的 final ppl
MAX_CROSS_RUN_HISTORY = 10   # prompt 里展示的历史 run 数量上限
SAVE_DIR        = f"/data/bulou/agent-d2z-discovery/results/agent_run{_SFX}"
# ──────────────────────────────────────────────────


class TrainingAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = AGENT_MODELS["controller"]   # 主模型/启动 ping 用 controller 档

        self.current_lr    = LR_INIT
        # ── 回合级超参（Strategist 在每回合开始决定，默认值=旧常量，向后兼容）──
        self.warmup_steps  = WARMUP_STEPS    # warmup 步数，0=不 warmup（Strategist 可关）
        self.decision_every = DECISION_EVERY # 决策频率 K（Strategist 每回合选）
        self.strategy_brief = None           # Memory 跨回合检索给 Controller 的经验
        self.loss_buffer   = []
        self.recent_raw_losses = []   # 每秒采集的 raw loss，最多保留 200 个点（密集采样）
        self.crash_count   = 0
        self.fast_crash_count = 0
        self.is_recovering = False
        # 当前 run 内的最近决策（短窗口，用于 prompt 的近因反馈）
        self.decision_history = []
        self.HISTORY_LEN = 30
        # 紧急触发器状态
        self.last_seen_loss = None       # 用于检测 loss 跳涨
        self.emergency_count = 0         # 当前 run 内紧急触发次数
        self.emergency_history = []      # [{step, prev_loss, curr_loss, lr_before, lr_after, reason}]
        # rollback 历史
        self.rollback_history = []       # [{crash_step, lr_at_crash, rolled_back_to_step, new_lr}]
        # Qwen 主动 rollback 请求（正常决策里输出的 [ROLLBACK_TO]）
        self.pending_rollback_step = None
        # 本 run 内总决策数（用于写 final_results_log）
        self.total_decisions_this_run = 0
        self.run_start_iso = datetime.datetime.now().isoformat()
        self.final_logged = False   # 幂等：本 run 是否已写过 final_results_log

        # 加载跨 run 历史（读 3 个 log 文件，聚合成 runs 列表）
        self.cross_run_history = self._load_cross_run_history()
        if self.cross_run_history:
            ppls = [r.get("final_ppl") for r in self.cross_run_history if r.get("final_ppl") is not None]
            print(
                f"[Agent] 加载跨 run 历史 {len(self.cross_run_history)} 条"
                f"（{len(ppls)} 条有 final_ppl，ppl 范围 "
                f"[{min(ppls):.2f}, {max(ppls):.2f}]）" if ppls else
                f"[Agent] 加载跨 run 历史 {len(self.cross_run_history)} 条",
                flush=True,
            )

        print(
            f"[Orchestrator] 启动 | task: {TOTAL_STEPS} steps | model={MODEL_CONFIG} | wandb={WANDB_PROJECT} | tag={RUN_TAG or '<none>'}\n"
            f"  固定配置: warmup={WARMUP_STEPS} | K={DECISION_EVERY} | init_lr={LR_INIT:.1e} | LR约束=[{LR_JUMP_LOW}x,{LR_JUMP_HIGH}x]当前\n"
            f"  2 Agent + 1 规则:\n"
            f"    [Agent] Controller={AGENT_MODELS['controller']}  —— 每K步调LR(loss+Memory检索)\n"
            f"    [Agent] Memory={AGENT_MODELS['memory']}  —— 跨回合策略检索器\n"
            f"    [Rules] SafetyGuard —— 崩溃回滚 + LR跳变约束(纯代码,无LLM)",
            flush=True,
        )
        self._ping_qwen()

    # ── 启动时 ping，确认 Qwen 真的可用 ──────────────
    def _ping_qwen(self):
        print(f"[Qwen Ping] 正在向 {self.model_name} 发送测试请求 ...", flush=True)
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "回复一个字：好"}],
                max_tokens=10,
                temperature=0.0,
            )
            dur = time.time() - t0
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            print(
                f"[Qwen Ping] ✓ 成功 | 耗时 {dur:.2f}s | 返回='{content.strip()}' | usage={usage}",
                flush=True,
            )
            with open(QWEN_LOG_FILE, "a") as f:
                f.write(
                    f"\n=== PING @ {datetime.datetime.now().isoformat()} ===\n"
                    f"model={self.model_name} dur={dur:.2f}s content={content!r} usage={usage}\n"
                )
        except Exception as e:
            print(f"[Qwen Ping] ✗ 失败: {e}", flush=True)
            print("[Qwen Ping] 请检查 DASHSCOPE_API_KEY 与模型名后再启动", flush=True)
            raise SystemExit(1)

    # ── 读取训练状态文件 ─────────────────────────────

    # ── 持久化 decision 日志 + 提取推理摘要 ──

    def _extract_reasoning(self, raw_response, max_len=120):
        """从 Qwen 完整输出里抽出推理那几句（去掉 [FINAL_LR] 标签那行）。"""
        lines = []
        for line in raw_response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if "FINAL_LR" in line or line.startswith("[") and line.endswith("]"):
                continue
            lines.append(line)
        text = " ".join(lines)
        if len(text) > max_len:
            text = text[:max_len] + "..."
        return text

    def _log_decision(self, step, loss, old_lr, new_lr, raw_lr_request, trend, reasoning):
        """把一次 Qwen 决策追加到 decisions_log.jsonl。"""
        entry = {
            "timestamp":      datetime.datetime.now().isoformat(),
            "run_start_iso":  self.run_start_iso,
            "step":           step,
            "loss":           round(float(loss), 4),
            "trend":          trend,
            "old_lr":         round(old_lr, 8),
            "new_lr":         round(new_lr, 8),
            "raw_lr_request": round(raw_lr_request, 8),
            "clamped":        abs(raw_lr_request - new_lr) > 1e-9,
            "reasoning":      reasoning,
        }
        try:
            with open(DECISIONS_LOG_FILE, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Agent] ⚠ 写 {DECISIONS_LOG_FILE} 失败: {e}", flush=True)

    # ── 持久化 rollback 日志（跨 run 累积） ──

    def _log_rollback(self, crash_step, lr_at_crash, rolled_back_to_step, new_lr, crash_reason="unknown"):
        """把一次 rollback 事件追加到 rollback_log.jsonl。"""
        entry = {
            "timestamp":           datetime.datetime.now().isoformat(),
            "run_start_iso":       self.run_start_iso,
            "crash_step":          crash_step,
            "lr_at_crash":         round(lr_at_crash, 8),
            "rolled_back_to_step": rolled_back_to_step,
            "new_lr":              round(new_lr, 8),
            "crash_reason":        crash_reason,
            "crash_count_in_run":  self.crash_count + 1,  # 本次 +1
        }
        try:
            with open(ROLLBACK_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(f"[Agent] ⚠ 写 {ROLLBACK_LOG_FILE} 失败: {e}", flush=True)

    # ── Self-healing 上下文：把崩溃 + 紧急介入历史摘要给 Qwen ──

    def _build_optimizer_specific_section(self):
        """根据 AGENT_MODE 加 optimizer-specific 行为约束 + 提示。"""
        mode = os.environ.get("AGENT_MODE", "cosine")
        if mode != "schedule_free":
            return ""
        return (
            "### Schedule-Free 模式特殊约束（必读） ###\n"
            f"本 run 用的不是普通 AdamW，而是 Schedule-Free AdamW。该 optimizer 内部维护 "
            f"averaged weights x（评估时实际部署的就是 x，不是当前训练 iterate y）。\n"
            f"**关键约束：你的 LR 输出必须 ≤ 当前 LR（{self.current_lr:.4e}）。**\n"
            f"原因：SF 内部把 lr_max ratchet 锁在历史最高值，averaging 权重 = lr_max²。"
            f"如果你提高 LR，lr_max 上升后永不回落，averaging 中所有过往步会被重新加权，"
            f"破坏 x 的收敛性（实测会让 final ppl 从 30 飙升到 45+）。\n"
            f"**正确策略：从 {LR_INIT:.2e} 起，单调下降，让 lr_max 锁定在初始 peak，"
            f"averaging 权重恒定，所有训练步均匀贡献到 x。**\n"
            f"任何 [FINAL_LR] > 当前 LR 都会被代码自动钳制为当前 LR。\n\n"
        )

    def _build_self_healing_section(self):
        """组装 self-healing 段：rollback 历史 + 紧急 LR 介入历史。"""
        if not self.rollback_history and not self.emergency_history:
            return ""

        lines = ["### Self-healing 历史 ###\n"]

        if self.rollback_history:
            lines.append(f"💥 本 run 已崩溃 + rollback {len(self.rollback_history)} 次：\n")
            for i, r in enumerate(self.rollback_history[-5:], 1):  # 只显示最近 5 次
                lines.append(
                    f"  #{i} step {r['crash_step']} 用 LR={r['lr_at_crash']:.2e} 训练时崩溃 → "
                    f"回退到 step {r['rolled_back_to_step']}（重启 LR={r['new_lr']:.2e}）\n"
                )

        if self.emergency_history:
            lines.append(f"🚨 本 run 已紧急介入 {len(self.emergency_history)} 次（loss 异常自动降 LR，未调 Qwen）：\n")
            for i, e in enumerate(self.emergency_history[-5:], 1):
                lines.append(
                    f"  #{i} step {e['step']} {e['reason']} → LR {e['lr_before']:.2e} → {e['lr_after']:.2e}\n"
                )

        return "".join(lines)

    # ── 紧急 self-healing：不调 Qwen，纯规则触发 ──────────────

    def _check_emergency(self, curr_step, curr_loss):
        """检测是否需要紧急介入。返回 (need_emergency: bool, reason: str)"""
        import math as _m
        # 1. NaN
        if _m.isnan(curr_loss) or _m.isinf(curr_loss):
            return True, f"loss={curr_loss}（NaN/Inf）"
        # 2. 绝对值过大（接近 random）
        if curr_loss > EMERGENCY_LOSS_MAX:
            return True, f"loss={curr_loss:.2f} 超过紧急阈值 {EMERGENCY_LOSS_MAX}"
        # 3. 跳涨过大
        if self.last_seen_loss is not None and self.last_seen_loss > 0:
            jump = curr_loss - self.last_seen_loss
            if jump > EMERGENCY_LOSS_JUMP:
                return True, f"loss 单次跳涨 {jump:+.2f}（阈值 {EMERGENCY_LOSS_JUMP}）"
        return False, ""

    def _emergency_lr_cut(self, step, prev_loss, curr_loss, reason):
        """紧急情况：直接砍 LR 不调 Qwen。"""
        old_lr = self.current_lr
        new_lr = max(LR_FLOOR, old_lr * EMERGENCY_LR_FACTOR)
        self.current_lr = new_lr
        # 原子写
        with open("lr_command_temp.txt", "w") as f:
            f.write(str(new_lr))
        os.replace("lr_command_temp.txt", LR_CMD_FILE)
        # 记录
        self.emergency_count += 1
        self.emergency_history.append({
            "step": step, "prev_loss": prev_loss, "curr_loss": curr_loss,
            "lr_before": old_lr, "lr_after": new_lr, "reason": reason,
        })
        print(
            f"[Agent] 🚨 EMERGENCY @ step {step} | {reason}\n"
            f"        LR {old_lr:.4e} → {new_lr:.4e}（×{EMERGENCY_LR_FACTOR}）",
            flush=True,
        )

    # ── 跨 run 历史加载 + 展示 ──

    def _load_cross_run_history(self):
        """读 final_results_log + decisions_log + rollback_log，聚合成 runs 列表。
        排除当前 run 自己（self.run_start_iso）。"""
        final_by_iso = {}
        if os.path.exists(FINAL_RESULTS_LOG_FILE):
            try:
                with open(FINAL_RESULTS_LOG_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            final_by_iso[r.get("run_start_iso")] = r
            except (OSError, json.JSONDecodeError):
                pass

        decisions_by_iso = {}
        if os.path.exists(DECISIONS_LOG_FILE):
            try:
                with open(DECISIONS_LOG_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        iso = d.get("run_start_iso")
                        if iso is None:
                            continue
                        decisions_by_iso.setdefault(iso, []).append(d)
            except OSError:
                pass

        rollbacks_by_iso = {}
        if os.path.exists(ROLLBACK_LOG_FILE):
            try:
                with open(ROLLBACK_LOG_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        iso = r.get("run_start_iso")
                        if iso is None:
                            continue
                        rollbacks_by_iso.setdefault(iso, []).append(r)
            except OSError:
                pass

        all_isos = set(final_by_iso) | set(decisions_by_iso) | set(rollbacks_by_iso)
        all_isos.discard(self.run_start_iso)   # 排除当前 run

        runs = []
        for iso in sorted(all_isos):
            fr = final_by_iso.get(iso, {})
            runs.append({
                "iso":       iso,
                "final_ppl": fr.get("final_eval_ppl"),
                # Strategist 的结构选择（Memory 学"结构→ppl"用）
                "warmup_steps":   fr.get("warmup_steps"),
                "decision_every": fr.get("decision_every"),
                "init_lr":        fr.get("init_lr"),
                "peak_lr":        fr.get("peak_lr_seen"),
                "min_lr":         fr.get("min_lr_seen"),
                "crash_count":    fr.get("crash_count"),
                "decisions": decisions_by_iso.get(iso, []),
                "rollbacks": rollbacks_by_iso.get(iso, []),
            })
        return runs

    def _build_cross_run_section(self):
        """把历史 runs 摘要成 prompt 段：完整 LR+loss 轨迹 + 显式标注历史最佳，
        引导 Qwen 主动超越历史最佳 ppl（方案 1：强化 in-context 跨 run 学习）。"""
        if not self.cross_run_history:
            return ""

        milestones = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

        def val_at_step(decisions, target_step, key):
            """返回 <= target_step 的最后一条决策里的 key 值（阶梯保持）。"""
            active = None
            for d in decisions:
                if d.get("step", 0) <= target_step:
                    active = d.get(key)
                else:
                    break
            return active

        # 历史最佳（final_ppl 最低）run —— 跨全部历史，不限于最近 N 条
        completed = [r for r in self.cross_run_history if r.get("final_ppl") is not None]
        best_run = min(completed, key=lambda r: r["final_ppl"]) if completed else None

        # 展示集合：最近 N 条 + 始终把历史最佳也带上（即便它较老）
        recent = self.cross_run_history[-MAX_CROSS_RUN_HISTORY:]
        show = list(recent)
        if best_run is not None and best_run not in show:
            show = [best_run] + show

        lines = ["### 历史 runs 记录（你的目标：用更优 LR 轨迹击败历史最佳 ppl） ###\n"]
        if best_run is not None:
            lines.append(
                f"🏆 历史最佳 ppl = {best_run['final_ppl']:.2f}"
                f"（{best_run['iso'][:19]}）。请分析它的 LR 形状为何更优，并尝试超越它。\n\n"
            )
        else:
            lines.append("（暂无已完成的历史 run，尚无 ppl 参考）\n\n")

        for i, run in enumerate(show, 1):
            iso_short = run["iso"][:19]
            ppl = run.get("final_ppl")
            decs = run["decisions"]
            rbs = run["rollbacks"]

            # 完整轨迹：每个 milestone 同时给 LR 和 当时的 train loss
            traj_parts = []
            for m in milestones:
                lr = val_at_step(decs, m, "new_lr")
                ls = val_at_step(decs, m, "loss")
                if lr is not None:
                    seg = f"s{m}:lr={lr:.2e}"
                    if ls is not None:
                        seg += f",L={ls:.2f}"
                    traj_parts.append(seg)
                else:
                    traj_parts.append(f"s{m}:?")
            traj = " | ".join(traj_parts)

            if ppl is not None:
                outcome = f"ppl={ppl:.2f}"
            elif rbs:
                last_crash = max(rbs, key=lambda r: r.get("crash_step", 0))
                outcome = f"未完成，crash@step {last_crash.get('crash_step')}"
            else:
                outcome = "未完成"

            star = " ★历史最佳" if (best_run is not None and run is best_run) else ""
            rb_note = f", {len(rbs)} 次 rollback" if rbs else ""
            lines.append(f"  #{i}{star} ({iso_short}, {outcome}{rb_note}):\n      {traj}\n")

        lines.append("\n")
        return "".join(lines)

    # ── 工具：列出当前可用的 ckpt step 列表 ──

    def _get_available_ckpt_steps(self):
        if not os.path.exists(SAVE_DIR):
            return []
        ckpts = sorted(
            [d for d in os.listdir(SAVE_DIR) if d.startswith("model_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        return [int(c.split("_")[-1]) for c in ckpts]

    # ── SafetyGuard 规则②用：选崩溃步前至少 ROLLBACK_BACK_STEPS 的最近健康 ckpt ──
    def _find_smart_checkpoint(self, crash_step):
        """
        崩溃后选回退 ckpt：选距离崩溃步至少 ROLLBACK_BACK_STEPS 步的最近 ckpt。
        这样能跳过"导致崩溃的那段训练"，避开同一个坏权重区域。
        """
        save_dir = SAVE_DIR
        if not os.path.exists(save_dir):
            return None, None
        ckpts = sorted(
            [d for d in os.listdir(save_dir) if d.startswith("model_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not ckpts:
            return None, None

        # 选崩溃步前至少 ROLLBACK_BACK_STEPS 步的最近 ckpt
        target_step = crash_step - ROLLBACK_BACK_STEPS
        valid = [c for c in ckpts if int(c.split("_")[-1]) <= max(0, target_step)]

        if valid:
            chosen = valid[-1]
            print(
                f"[Agent] 智能回退：崩溃@step {crash_step}，"
                f"选 {chosen}（距离崩溃 {crash_step - int(chosen.split('_')[-1])} 步）",
                flush=True,
            )
        elif ckpts:
            # 没有足够远的 ckpt，用最早那个
            chosen = ckpts[0]
            print(f"[Agent] 智能回退：无足够远 ckpt，退到最早 {chosen}", flush=True)
        else:
            return None, None

        chosen_step = int(chosen.split("_")[-1])
        return os.path.join(save_dir, chosen), chosen_step

    def _read_state(self, retries=3, retry_delay=0.2):
        for attempt in range(retries):
            try:
                with open(STATE_FILE) as f:
                    content = f.read().strip()
                if content:
                    return json.loads(content)
                if attempt < retries - 1:
                    time.sleep(retry_delay)
            except FileNotFoundError:
                return None
            except (json.JSONDecodeError, OSError):
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        return None

    # ── Qwen API 决策 LR ────────────────────────────

    def get_decision(self, state):
        step         = state.get("step", 0)
        progress     = state.get("progress_percent", 0)
        current_loss = state.get("loss", 0)

        if current_loss > 0:
            self.loss_buffer.append(round(float(current_loss), 4))
        if len(self.loss_buffer) > 30:
            self.loss_buffer.pop(0)

        trend_desc = "数据不足"
        if len(self.loss_buffer) >= 3:
            delta = self.loss_buffer[-1] - self.loss_buffer[0]
            if delta < -0.05:
                trend_desc = f"下降 {delta:+.3f}"
            elif delta > 0.05:
                trend_desc = f"上升 {delta:+.3f}"
            else:
                trend_desc = f"平稳 {delta:+.3f}"

        # 组装历史决策轨迹，方便 Qwen 看到自己之前操作的效果
        if self.decision_history:
            history_lines = []
            for i, h in enumerate(self.decision_history):
                history_lines.append(
                    f"  #{i+1} step={h['step']} loss={h['loss']:.4f} "
                    f"LR: {h['old_lr']:.2e} → {h['new_lr']:.2e}"
                )
            history_desc = "\n".join(history_lines)
            # 最近一条决策到现在的 loss 变化，用于自我评估
            last = self.decision_history[-1]
            if self.loss_buffer:
                effect = self.loss_buffer[-1] - last["loss"]
                effect_desc = (
                    f"自上次决策 (step {last['step']}, LR {last['new_lr']:.2e}) 以来，"
                    f"loss 变化 {effect:+.4f}"
                )
            else:
                effect_desc = "尚无新 loss 数据"
        else:
            history_desc = "  （无，本次为首次决策）"
            effect_desc = "—"

        prompt = (
            f"你正在调一个训练系统的控制参数 LR（learning rate）。任务总长 {TOTAL_STEPS} 步，目标：让最终 eval loss 最小。\n\n"

            f"### 训练动力学（中性事实） ###\n"
            f"1. LR 决定 AdamW 每步的有效更新量。LR 大→步子大；LR 小→步子小。\n"
            f"2. 梯度信噪比随训练阶段变化：早期梯度方向信息丰富但方差大；后期梯度幅值变小、噪声占比上升。\n"
            f"3. LR 序列在每 {self.decision_every} 步独立可选，形状完全自由。\n"
            f"4. 你不知道当前 loss 地形长什么样，必须从 loss 反馈推断。\n\n"

            f"### 任务参数 ###\n"
            f"- 起步：本回合 warmup={self.warmup_steps} 步（0=无 warmup，直接从初始 LR 开跑），仅防发散、非最优值\n"
            f"- 你的控制区间：**warmup 之后 → {TOTAL_STEPS}**，peak 多高、何时衰减、什么形状全部由你自行探索决定\n"
            f"- LR 范围：[{LR_FLOOR:.0e}, {LR_CEIL:.0e}]\n"
            f"- **每次调整幅度受限**：新 LR 会被钳制在当前值的 [{LR_JUMP_LOW}×, {LR_JUMP_HIGH}×]（SafetyGuard 硬约束），所以一步最多翻倍/减半，要平滑变化\n\n"

            f"### 当前训练状态 ###\n"
            f"- Step：{step} / {TOTAL_STEPS}（{progress:.2f}%）\n"
            f"- 当前 LR：{self.current_lr:.4e}\n"
            f"- 近期 Loss（时间正序）：{self.loss_buffer}\n"
            f"- 近 200 步密集 loss 采样（每 10 步一个点）：{self.recent_raw_losses[::10]}\n"
            f"- Loss 趋势：{trend_desc}\n"
            f"- 当前 run 累计崩溃：{self.crash_count} 次\n\n"

            f"### 当前 run 内决策历史（最近 {self.HISTORY_LEN} 条） ###\n"
            f"{history_desc}\n"
            f"上次决策的 loss 效果：{effect_desc}\n\n"

            f"{self._build_self_healing_section()}"

            f"{self._memory_brief_section()}"

            f"{self._build_optimizer_specific_section()}"

            f"### 任务 ###\n"
            f"基于当前 loss 反馈和跨回合经验，决定本 step 的 LR。\n"
            f"先简要说明（≤3 句）你的推理逻辑，然后严格输出标签：\n"
            f"[FINAL_LR] <数值> [/FINAL_LR]\n"
            f"示例：[FINAL_LR] 1.5e-3 [/FINAL_LR]\n"
            f"（崩溃回滚已交给 SafetyGuard 自动处理，你不需要关心。）"
        )

        api_t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=AGENT_MODELS["controller"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5,   # 温和探索：比 0.3 多样，比 0.7 稳定
            )
            api_dur = time.time() - api_t0
            res = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            print(
                f"[Qwen Call] step={step} | 耗时 {api_dur:.2f}s | 模型={self.model_name} | usage={usage}",
                flush=True,
            )
            try:
                with open(QWEN_LOG_FILE, "a") as f:
                    f.write(
                        f"\n=== CALL @ {datetime.datetime.now().isoformat()} step={step} ===\n"
                        f"dur={api_dur:.2f}s usage={usage}\n"
                        f"--- prompt ---\n{prompt}\n"
                        f"--- response ---\n{res}\n"
                    )
            except Exception:
                pass
        except Exception as e:
            api_dur = time.time() - api_t0
            print(f"[Qwen Call] ✗ 失败 (耗时 {api_dur:.2f}s): {e}，保持当前 LR {self.current_lr:.4e}", flush=True)
            return self.current_lr

        # 三级解析
        match = re.search(r"\[FINAL_LR\]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\[/FINAL_LR\]", res)
        if not match:
            match = re.search(r"FINAL_LR:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", res)
        if not match:
            match = re.search(r"\b([0-9]*\.?[0-9]+[eE][+-]?[0-9]+)\b", res)

        if match:
            new_lr = float(match.group(1))
            print(f"\n[Qwen 输出]\n{res.strip()}\n", flush=True)

            clamped = max(LR_FLOOR, min(LR_CEIL, new_lr))
            if clamped != new_lr:
                print(f"[Agent] 限幅: {new_lr:.4e} → {clamped:.4e}", flush=True)

            # SF 模式：强制 monotone-decrease，防止 lr_max ratchet 破坏 averaging 权重
            if os.environ.get("AGENT_MODE") == "schedule_free":
                if clamped > self.current_lr:
                    print(
                        f"[Agent] SF monotone 约束: {clamped:.4e} > 当前 {self.current_lr:.4e}, 钳制为当前值",
                        flush=True,
                    )
                    clamped = self.current_lr

            # ── 解析可选的 [ROLLBACK_TO] 标签 ──
            rollback_m = re.search(r"\[ROLLBACK_TO\]\s*(\d+)\s*\[/ROLLBACK_TO\]", res)
            if rollback_m:
                requested_step = int(rollback_m.group(1))
                available = self._get_available_ckpt_steps()
                if available:
                    if requested_step in available:
                        self.pending_rollback_step = requested_step
                        print(f"[Agent] 🔄 Qwen 主动 rollback 请求 → step {requested_step}", flush=True)
                    else:
                        closest = min(available, key=lambda s: abs(s - requested_step))
                        self.pending_rollback_step = closest
                        print(
                            f"[Agent] 🔄 Qwen 请求 rollback 到 step {requested_step} 不存在，用最近 {closest}",
                            flush=True,
                        )
                else:
                    print(f"[Agent] ⚠ Qwen 请求 rollback 但无可用 ckpt，忽略", flush=True)

            self.decision_history.append({
                "step":   step,
                "loss":   float(current_loss),
                "old_lr": self.current_lr,
                "new_lr": clamped,
            })
            if len(self.decision_history) > self.HISTORY_LEN:
                self.decision_history.pop(0)

            self.total_decisions_this_run += 1

            # ── 一行汇总（人类友好） ──
            arrow = "→" if abs(clamped - self.current_lr) > 1e-9 else "·"
            change_pct = ((clamped - self.current_lr) / self.current_lr * 100) if self.current_lr > 0 else 0
            print(
                f"╔══ Qwen 决策 #{len(self.decision_history)} @ step {step} ══╗\n"
                f"║ Loss : {current_loss:.4f}  趋势：{trend_desc}\n"
                f"║ LR   : {self.current_lr:.4e} {arrow} {clamped:.4e}  ({change_pct:+.1f}%)\n"
                f"║ 推理 : {self._extract_reasoning(res)}\n"
                f"╚══════════════════════════════════════╝",
                flush=True,
            )

            # ── 持久化到 decisions_log.jsonl ──
            self._log_decision(
                step=step, loss=current_loss,
                old_lr=self.current_lr, new_lr=clamped,
                raw_lr_request=new_lr,
                trend=trend_desc,
                reasoning=res.strip(),
            )

            return clamped

        print(f"[Agent] 解析失败，保持当前 LR {self.current_lr:.4e}\n原始输出:\n{res.strip()}", flush=True)
        return self.current_lr

    # ── 启动训练子进程 ───────────────────────────────

    def _find_latest_checkpoint(self, after_crash=False):
        """
        找 /data/bulou/agent-d2z-discovery/results/agent_run 下可用的 checkpoint。
        after_crash=True 时跳过最新的（可能损坏），回到倒数第二个。
        """
        save_dir = "/data/bulou/agent-d2z-discovery/results/agent_run"
        if not os.path.exists(save_dir):
            return None
        ckpts = sorted(
            [d for d in os.listdir(save_dir) if d.startswith("model_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not ckpts:
            return None

        if after_crash and len(ckpts) >= 2:
            # 跳过最新的（可能是坏档），回到倒数第二个
            ckpt_path = os.path.join(save_dir, ckpts[-2])
            print(f"[Agent] 炸档回退，跳过 {ckpts[-1]}，使用: {ckpt_path}", flush=True)
        else:
            ckpt_path = os.path.join(save_dir, ckpts[-1])
            print(f"[Agent] 找到 checkpoint: {ckpt_path}", flush=True)

        return ckpt_path

    def _reader_thread(self, pipe, q):
        try:
            for line in iter(pipe.readline, ""):
                q.put(line)
        finally:
            pipe.close()

    def _drain_queue(self, q, timeout=0.05):
        while True:
            try:
                line = q.get(timeout=timeout)
                print(line.rstrip("\n"), flush=True)
            except queue.Empty:
                break

    def _launch_train(self, lr, continue_from=None):
        # 自适应：DEVICE="7" → 单卡（python main.py）；DEVICE="0,1,2,3" → torchrun。
        device = os.environ.get("DEVICE", "0")
        port   = os.environ.get("MASTER_PORT", "29500")
        nproc  = len([d for d in device.split(",") if d.strip()])

        # 通过 env 切换 baseline 模式 (cosine + bf16) 或 combined 模式 (SF + fp32)
        # AGENT_MODE=schedule_free → 组合模式
        agent_mode = os.environ.get("AGENT_MODE", "cosine")
        if agent_mode == "schedule_free":
            optimizer_type = "schedule_free"
            dtype = "float32"   # SF + bf16 会把 averaged weights 弄烂 → eval ppl 发散
        else:
            optimizer_type = "adamw"
            dtype = "bfloat16"

        train_args = [
            "main.py",
            "--model_type",        "llama",
            "--model_config",      MODEL_CONFIG,
            "--lr",                str(lr),
            "--optimizer",         "adamw",
            "--optimizer_type",    optimizer_type,
            "--batch_size",        "64",
            "--total_batch_size",  "512",
            "--num_training_steps", str(TOTAL_STEPS),
            "--warmup_steps",       str(self.warmup_steps),
            "--weight_decay",      "0.01",
            "--dtype",             dtype,
            "--eval_every",        "1000",
            "--save_every",        str(self.decision_every),   # ckpt 与决策点对齐：每个 LR 决策处都存，Guardian 可回滚到任意健康决策点（回合结束 clean_run 全删）
            "--grad_clipping",     "1.0",
            "--save_dir",          SAVE_DIR,
            "--wandb_project",     WANDB_PROJECT,
        ]

        if continue_from:
            train_args += ["--continue_from", continue_from]
            print(f"[Agent] 从 checkpoint 续跑: {continue_from}", flush=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device

        if nproc <= 1:
            # 单卡：直接 python，自己注入分布式环境变量
            cmd = ["python"] + train_args
            env["LOCAL_RANK"]  = "0"
            env["RANK"]        = "0"
            env["WORLD_SIZE"]  = "1"
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = port
            print(f"[Launch] 单卡模式 (DEVICE={device})", flush=True)
        else:
            # 多卡：torchrun 自动注入 RANK/LOCAL_RANK/WORLD_SIZE
            cmd = [
                "torchrun", "--standalone",
                f"--nproc-per-node={nproc}",
                f"--master-port={port}",
            ] + train_args
            print(f"[Launch] 多卡模式 nproc={nproc} (DEVICE={device})", flush=True)

        print(f"[Launch] {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        print(f"[Launch] PID={proc.pid}", flush=True)

        q = queue.Queue()
        t = threading.Thread(target=self._reader_thread, args=(proc.stdout, q))
        t.daemon = True
        t.start()
        return proc, q

    # ── 写 final_results_log（跨 run 学习用，两条完成路径共用）──
    def _write_final_result(self):
        """读 main.py 写的 {SAVE_DIR}/final_result.json，追加到 final_results_log.jsonl。
        幂等：本 run 只写一次。无论训练是经 progress>=99.9 还是 exit_code==0 结束都会调用。"""
        if self.final_logged:
            return
        final_path = os.path.join(SAVE_DIR, "final_result.json")
        final_data = None
        for _ in range(120):   # 等 main.py 写 final_result.json，最多 60s
            if os.path.exists(final_path):
                try:
                    with open(final_path) as _f:
                        final_data = json.load(_f)
                    break
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.5)
        if final_data is None:
            print(f"[Agent] ⚠ 未读到 {final_path}，本 run 不写入 final_results_log", flush=True)
            return
        entry = {
            "timestamp":       datetime.datetime.now().isoformat(),
            "run_start_iso":   self.run_start_iso,
            "final_eval_ppl":  final_data.get("final_eval_ppl"),
            "final_eval_loss": final_data.get("final_eval_loss"),
            "total_decisions": self.total_decisions_this_run,
            "crash_count":     self.crash_count,
            "peak_lr_seen":    max([d["new_lr"] for d in self.decision_history], default=None),
            "min_lr_seen":     min([d["new_lr"] for d in self.decision_history], default=None),
            # Strategist 本回合的结构选择（Memory 学"结构→ppl"用）
            "warmup_steps":    self.warmup_steps,
            "decision_every":  self.decision_every,
            "init_lr":         LR_INIT,
            "optimizer":       "adamw",
        }
        try:
            with open(FINAL_RESULTS_LOG_FILE, "a") as _f:
                _f.write(json.dumps(entry) + "\n")
            self.final_logged = True
            ppl = entry["final_eval_ppl"]
            print(
                f"[Agent] ✓ 写入 {FINAL_RESULTS_LOG_FILE}: ppl={ppl:.2f}"
                if ppl is not None else f"[Agent] ✓ 写入 {FINAL_RESULTS_LOG_FILE}",
                flush=True,
            )
        except OSError as _e:
            print(f"[Agent] ⚠ 写 {FINAL_RESULTS_LOG_FILE} 失败: {_e}", flush=True)

    def _memory_brief_section(self):
        """Controller prompt 用：注入 Memory 蒸馏的跨回合经验（取代原始历史 dump）。"""
        if not self.strategy_brief:
            return ""
        return (
            "### 跨回合经验（记忆模块蒸馏，来自本实验自身历史、非外部答案） ###\n"
            f"{self.strategy_brief}\n\n"
        )

    # ── Memory 子 agent：把历史回合蒸馏成探索建议（ckpt 删除后唯一的跨回合记忆）──
    def _memory_brief(self):
        """读本 agent 自己跑过的历史回合（结构超参 + LR 轨迹 + 最终 ppl），
        调 LLM 蒸馏出：①与低 ppl 相关的规律 ②下一回合的具体探索建议。
        关键：只基于自己试过的结果推理，禁止引入外部已知答案（cosine/D2Z 名字或推荐值）。
        结果存 self.strategy_brief，供 Controller 用。无历史则置 None。"""
        completed = [r for r in self.cross_run_history if r.get("final_ppl") is not None]
        if not completed:
            self.strategy_brief = None
            print("[Memory] 无已完成历史，本回合自由探索", flush=True)
            return

        milestones = [1000, 3000, 5000, 7000, 9000]
        def lr_at(decs, m):
            v = None
            for d in decs:
                if d.get("step", 0) <= m: v = d.get("new_lr")
                else: break
            return v
        recent = self.cross_run_history[-MAX_CROSS_RUN_HISTORY:]
        lines = []
        for r in recent:
            ppl = r.get("final_ppl")
            traj = " ".join(f"s{m}:{(lr_at(r['decisions'],m) or 0):.1e}" for m in milestones)
            lines.append(
                f"warmup={r.get('warmup_steps')} K={r.get('decision_every')} "
                f"init_lr={r.get('init_lr')} peak={r.get('peak_lr')} min={r.get('min_lr')} "
                f"crash={r.get('crash_count')} → ppl={'%.2f'%ppl if ppl else '未完成'} | LR轨迹 {traj}"
            )
        table = "\n".join(lines)
        prompt = (
            f"你是训练策略实验的『记忆模块』。下面是本实验自己跑过的历史回合（LR 轨迹 + 最终 ppl，ppl 越低越好）：\n"
            f"{table}\n\n"
            f"⚠️ 重要约束：warmup、K、初始 LR 在所有回合都**固定不可改**，你的建议只能影响 Controller "
            f"对每 K 步 LR 的决策（即 LR 轨迹的形状）。**不要建议改 warmup/K/初始 LR**，那超出了能控范围。\n\n"
            f"### Plateau 检测（强制规则） ###\n"
            f"扫描上表里的近 **10** 条历史回合（不足 10 条则跳过本规则），若同时满足：\n"
            f"  (a) 它们的 peak_lr 的 max/min 比值 < 2（peak 集中在同一数量级）\n"
            f"  (b) 它们的 final_ppl 极差在均值 ±10% 以内（无显著改进）\n"
            f"则判定为 **plateau**，必须在 brief 里**强烈建议** Controller 下一回合探索**与历史 peak 数量级跨度不同的区域**"
            f"（往高 1 个或低 1 个数量级方向走，由你根据数据判断方向）。这是 multi-armed bandit 探索范式，**不依赖外部最优答案**。\n"
            f"若未触发 plateau（包括样本数 <10），正常归纳即可。\n\n"
            f"### 任务 ###\n"
            f"基于**这些数据本身**推理（不要引入任何外部已知的调度方法名或推荐数值，只从数据里归纳）：\n"
            f"1. 历史回合 LR 轨迹的形状（峰值高低、衰减时机、末段是否失速）与最终 ppl 有什么关联？\n"
            f"2. 给下一回合的 Controller 一条**具体、可操作**的轨迹建议：LR 在训练早/中/后期分别该维持/上探/下衰？大致形状是什么样？若 plateau 触发，明确指出探索方向。\n"
            f"≤150 字，直接给结论。"
        )
        try:
            resp = self.client.chat.completions.create(
                model=AGENT_MODELS["memory"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0.4,
            )
            self.strategy_brief = resp.choices[0].message.content.strip()
            try:
                with open(QWEN_LOG_FILE, "a") as f:
                    f.write(f"\n=== MEMORY @ {datetime.datetime.now().isoformat()} ===\n"
                            f"--- prompt ---\n{prompt}\n--- brief ---\n{self.strategy_brief}\n")
            except Exception:
                pass
            print(f"[Memory] 经验简报：{self.strategy_brief[:120]}…", flush=True)
        except Exception as e:
            self.strategy_brief = None
            print(f"[Memory] ✗ 蒸馏失败 ({e})，本回合无经验先验", flush=True)

    # ── Strategist 子 agent：回合开始决定结构超参 ──────────────
    # ── SafetyGuard 规则①：LR 跳变约束（纯代码，无 LLM）──────────────
    def _safetyguard_clamp(self, old_lr, proposed_lr):
        """把 Controller 提议的 LR 钳制在 [LR_JUMP_LOW×, LR_JUMP_HIGH×] 当前值，再夹到全局 [FLOOR, CEIL]。"""
        lo = LR_JUMP_LOW * old_lr
        hi = LR_JUMP_HIGH * old_lr
        clamped = max(lo, min(hi, proposed_lr))
        clamped = max(LR_FLOOR, min(LR_CEIL, clamped))
        if abs(clamped - proposed_lr) > 1e-12:
            print(f"[SafetyGuard] LR 跳变约束: 提议 {proposed_lr:.3e} → 钳制 {clamped:.3e} "
                  f"(允许区间 [{lo:.3e}, {hi:.3e}])", flush=True)
        return clamped

    # ── 主循环 ──────────────────────────────────────

    def run(self):
        for f in [LR_CMD_FILE, STATE_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # 启动时自动找最新 checkpoint 续跑
        latest_ckpt = self._find_latest_checkpoint()
        # 全新回合：Memory 跨回合检索经验，喂给 Controller（warmup/K/初始LR 现为固定配置，无 Strategist）
        if latest_ckpt is None:
            self._memory_brief()
        proc, out_queue = self._launch_train(self.current_lr, continue_from=latest_ckpt)
        start_run_time  = time.time()

        last_decision_step = 0
        last_seen_step     = 0
        last_seen_curr_step = 0   # 记录最后看到的 step（crash 时用来定位）

        while True:
            self._drain_queue(out_queue)

            # 子进程退出处理
            if proc.poll() is not None:
                time.sleep(0.5)
                self._drain_queue(out_queue)

                run_duration = time.time() - start_run_time
                exit_code    = proc.returncode
                print(f"\n[Agent] 进程退出 | 存活: {run_duration:.1f}s | exit_code={exit_code}", flush=True)

                # ── 区分「正常完成」与「真 crash」：exit_code==0 是正常退出
                if exit_code == 0:
                    # 再读一次 state 确认 progress
                    final_state = self._read_state()
                    final_progress = final_state.get("progress_percent", 0) if final_state else 0
                    print(
                        f"[Agent] ✓ 本次 run 正常完成 (exit_code=0, progress={final_progress:.1f}%)",
                        flush=True,
                    )
                    self._write_final_result()
                    break

                is_fast_crash = run_duration < 60
                self.fast_crash_count = self.fast_crash_count + 1 if is_fast_crash else 0

                # ── 决定是否放弃本次 run（达到 MAX_FAST_CRASH 上限）──
                if self.fast_crash_count >= MAX_FAST_CRASH:
                    print(
                        f"[Agent] 💥 连续快速崩溃 {MAX_FAST_CRASH} 次，判定本次 run 失败，退出。",
                        flush=True,
                    )
                    return  # 整个 agent.py 退出，外层 for 循环会启动新 run

                # 单次崩溃：智能回退 + 自动降 LR + 记录 rollback 历史
                crash_wait = min(10 * max(self.fast_crash_count, 1), 60)
                time.sleep(crash_wait)

                lr_at_crash = self.current_lr
                # ── SafetyGuard 规则②：崩溃回滚（纯代码，无 LLM）──
                # 选崩溃步前至少 ROLLBACK_BACK_STEPS 的最近健康 ckpt，LR 自动砍半防再炸。
                latest_ckpt, ckpt_step = self._find_smart_checkpoint(last_seen_curr_step)
                ckpt_step = ckpt_step or 0
                self.current_lr = max(LR_FLOOR, lr_at_crash * ROLLBACK_LR_FACTOR)
                print(
                    f"[SafetyGuard] 崩溃回滚 → ckpt step {ckpt_step} | "
                    f"LR {lr_at_crash:.3e}→{self.current_lr:.3e} (×{ROLLBACK_LR_FACTOR})",
                    flush=True,
                )

                # 记录到 rollback_history（给后续 prompt 用）
                self.rollback_history.append({
                    "crash_step":    last_seen_curr_step,
                    "lr_at_crash":   lr_at_crash,
                    "rolled_back_to_step": ckpt_step,
                    "new_lr":        self.current_lr,
                })
                # 持久化到磁盘（跨 run 累积）
                self._log_rollback(
                    crash_step=last_seen_curr_step,
                    lr_at_crash=lr_at_crash,
                    rolled_back_to_step=ckpt_step,
                    new_lr=self.current_lr,
                    crash_reason=f"exit_code_{exit_code}",
                )

                self.crash_count   += 1
                self.loss_buffer    = []
                self.recent_raw_losses = []
                self.last_seen_loss = None    # 重置紧急检测器
                self.is_recovering  = True
                last_decision_step  = 0
                last_seen_step      = 0

                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)

                print(
                    f"[Agent] 💥 崩溃 #{self.crash_count} | crash@step {last_seen_curr_step} "
                    f"with LR={lr_at_crash:.2e}\n"
                    f"        → SafetyGuard 规则：回退到 step {ckpt_step}，重启 LR={self.current_lr:.2e}",
                    flush=True,
                )
                start_run_time = time.time()
                proc, out_queue = self._launch_train(self.current_lr, continue_from=latest_ckpt)
                continue

            state = self._read_state()
            if state is None:
                time.sleep(1)
                continue

            # 训练完成
            if state.get("progress_percent", 0) >= 99.9:
                print("[Agent] 训练完成，等待 main.py 退出...", flush=True)
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    print("[Agent] 主进程 10min 内未退出，强制终止", flush=True)
                    proc.terminate()
                self._drain_queue(out_queue)
                self._write_final_result()   # 经 progress>=99.9 完成也要写 final_results_log（跨 run 记忆依赖它）
                print("[Agent] ✓ 本次 run 完成", flush=True)
                break

            curr_step = state.get("step", 0)
            curr_loss = state.get("loss", 0)
            last_seen_curr_step = curr_step

            # ── 异常检测 → 由 SafetyGuard 决定是否触发回滚 ──
            import math as _m
            if curr_loss > 0 or _m.isnan(curr_loss) or _m.isinf(curr_loss):
                need_emergency, reason = self._check_emergency(curr_step, curr_loss)
                if need_emergency:
                    self.emergency_history.append({
                        "step": curr_step, "prev_loss": self.last_seen_loss,
                        "curr_loss": curr_loss, "lr_before": self.current_lr,
                        "lr_after": self.current_lr, "reason": reason,
                    })
                    self.emergency_count += 1
                    # ── SafetyGuard 规则③：NaN/Inf 或 loss 超阈值 → 终止子进程触发崩溃回滚 ──
                    # (跳涨 spike 是噪声，只记录不杀；NaN/灾难是真异常，必须 rollback)
                    catastrophic = (_m.isnan(curr_loss) or _m.isinf(curr_loss)
                                    or curr_loss > EMERGENCY_LOSS_MAX)
                    if catastrophic:
                        print(
                            f"[SafetyGuard] 🚨 严重异常 @ step {curr_step}: {reason}\n"
                            f"            → 终止子进程，触发崩溃回滚（rule-based）",
                            flush=True,
                        )
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        # 让下一次循环 proc.poll() != None 接管 SafetyGuard rollback
                        continue
                    else:
                        print(
                            f"[Agent] ⚠ loss 跳涨 @ step {curr_step}: {reason}（仅记录，非灾难）",
                            flush=True,
                        )
                self.last_seen_loss = curr_loss
                # 密集采样 loss（每秒一次），最多保留 200 个点
                self.recent_raw_losses.append(round(float(curr_loss), 4))
                if len(self.recent_raw_losses) > 200:
                    self.recent_raw_losses.pop(0)

            if self.is_recovering and curr_step > 0:
                last_decision_step = curr_step - self.decision_every
                self.is_recovering = False
                print(f"[Agent] 复活于 Step {curr_step}，准备决策", flush=True)

            step_changed   = (curr_step != last_seen_step)
            last_seen_step = curr_step

            # Controller 只在 warmup 之后接管（warmup 由 Strategist 设、main.py 的 scheduler 跑）；
            # warmup 期间不写 lr_command，避免覆盖 warmup 斜坡。warmup=0 时从首个 K 触发。
            should_trigger = (
                curr_step >= self.warmup_steps
                and curr_step >= last_decision_step + self.decision_every
            )

            if step_changed:
                status = "触发决策 ✓" if should_trigger else f"等待（需 >= {last_decision_step + self.decision_every}）"
                print(f"[Debug] step={curr_step} | {status}", flush=True)

            if should_trigger:
                old_lr             = self.current_lr
                proposed_lr        = self.get_decision(state)        # Controller 提议
                self.current_lr    = self._safetyguard_clamp(old_lr, proposed_lr)  # SafetyGuard 跳变约束(纯规则)
                last_decision_step = curr_step

                with open("lr_command_temp.txt", "w") as f:
                    f.write(str(self.current_lr))
                os.replace("lr_command_temp.txt", LR_CMD_FILE)
                print(f"[Agent] LR 指令写入: {self.current_lr:.4e}", flush=True)

                # ── Qwen 主动 rollback 请求处理 ──
                if self.pending_rollback_step is not None:
                    target_step = self.pending_rollback_step
                    self.pending_rollback_step = None
                    lr_for_restart = self.current_lr

                    print(
                        f"[Agent] 🔄 执行主动 rollback：终止当前训练 @ step {curr_step} → "
                        f"回到 step {target_step}，重启 LR={lr_for_restart:.2e}",
                        flush=True,
                    )

                    # 终止当前子进程
                    try:
                        proc.terminate()
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    self._drain_queue(out_queue)

                    ckpt_path = os.path.join(SAVE_DIR, f"model_{target_step}")
                    if not os.path.exists(ckpt_path):
                        # 理论上 pending_rollback_step 已被验证，兜底处理
                        available = self._get_available_ckpt_steps()
                        if available:
                            target_step = min(available, key=lambda s: abs(s - target_step))
                            ckpt_path = os.path.join(SAVE_DIR, f"model_{target_step}")
                        else:
                            ckpt_path = None

                    # 记录到 rollback_history + 持久化
                    self.rollback_history.append({
                        "crash_step":    curr_step,
                        "lr_at_crash":   lr_for_restart,
                        "rolled_back_to_step": target_step,
                        "new_lr":        lr_for_restart,
                    })
                    self._log_rollback(
                        crash_step=curr_step,
                        lr_at_crash=lr_for_restart,
                        rolled_back_to_step=target_step,
                        new_lr=lr_for_restart,
                        crash_reason="qwen_proactive_rollback",
                    )

                    # 重置状态
                    self.crash_count      += 1
                    self.loss_buffer       = []
                    self.recent_raw_losses = []
                    self.last_seen_loss    = None
                    self.is_recovering     = True
                    last_decision_step     = 0
                    last_seen_step         = 0
                    last_seen_curr_step    = 0

                    if os.path.exists(STATE_FILE):
                        os.remove(STATE_FILE)

                    # 重启训练子进程
                    start_run_time = time.time()
                    proc, out_queue = self._launch_train(self.current_lr, continue_from=ckpt_path)
                    continue

            time.sleep(1)


if __name__ == "__main__":
    TrainingAgent().run()
