from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DPO_TRACE = ROOT / "outputs" / "retrieval_policy_dpo_torch" / "training_trace.json"
PPO_TRACE = ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "training_trace.json"
OUT = ROOT / "docs" / "rl_training_convergence.svg"


def read_json(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def moving_average(values: list[float], window: int) -> list[float]:
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        smoothed.append(mean(values[start : index + 1]))
    return smoothed


def sample_xy(xs: list[float], ys: list[float], max_points: int = 120) -> tuple[list[float], list[float]]:
    if len(xs) <= max_points:
        return xs, ys
    step = (len(xs) - 1) / (max_points - 1)
    indexes = sorted({round(i * step) for i in range(max_points)})
    return [xs[i] for i in indexes], [ys[i] for i in indexes]


def nice_range(values: list[float], pad_ratio: float = 0.08) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        return low - 1, high + 1
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def scale_points(
    xs: list[float],
    ys: list[float],
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    left, top, width, height = box
    x_min, x_max = x_range
    y_min, y_max = y_range
    points: list[str] = []
    for x_value, y_value in zip(xs, ys):
        x = left + (x_value - x_min) / (x_max - x_min) * width
        y = top + height - (y_value - y_min) / (y_max - y_min) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def axis_labels(
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_ticks: int = 5,
    y_ticks: int = 5,
) -> str:
    left, top, width, height = box
    x_min, x_max = x_range
    y_min, y_max = y_range
    parts: list[str] = []
    for i in range(x_ticks):
        ratio = i / (x_ticks - 1)
        x = left + ratio * width
        value = x_min + ratio * (x_max - x_min)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + height + 28}" text-anchor="middle" class="tick">{value:.0f}</text>')
    for i in range(y_ticks):
        ratio = i / (y_ticks - 1)
        y = top + height - ratio * height
        value = y_min + ratio * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    return "\n".join(parts)


def legend(items: list[tuple[str, str]], x: int, y: int) -> str:
    parts: list[str] = []
    offset = 0
    for color, label in items:
        parts.append(f'<line x1="{x + offset}" y1="{y}" x2="{x + offset + 28}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        parts.append(f'<text x="{x + offset + 38}" y="{y + 5}" class="legend">{label}</text>')
        offset += 165
    return "\n".join(parts)


def main() -> None:
    dpo = read_json(DPO_TRACE)
    ppo = read_json(PPO_TRACE)

    dpo_epochs = [float(row["epoch"]) for row in dpo]
    total_loss = [float(row["loss"]) for row in dpo]
    dpo_loss = [float(row["dpo_loss"]) for row in dpo]
    sft_loss = [float(row["sft_loss"]) for row in dpo]

    ppo_episodes = [float(row["episode"]) for row in ppo]
    ppo_rewards = moving_average([float(row["reward"]) for row in ppo], window=12)
    ppo_losses = moving_average([float(row["loss"]) for row in ppo], window=12)
    ppo_episodes, ppo_rewards = sample_xy(ppo_episodes, ppo_rewards)
    ppo_loss_x, ppo_losses = sample_xy([float(row["episode"]) for row in ppo], ppo_losses)

    ppo_box = (92, 112, 470, 238)
    dpo_box = (690, 112, 470, 238)
    ppo_x_range = (min(ppo_episodes), max(ppo_episodes))
    ppo_y_range = nice_range(ppo_rewards + ppo_losses)
    dpo_x_range = (min(dpo_epochs), max(dpo_epochs))
    dpo_y_range = nice_range(total_loss + dpo_loss + sft_loss)

    ppo_reward_points = scale_points(ppo_episodes, ppo_rewards, ppo_box, ppo_x_range, ppo_y_range)
    ppo_loss_points = scale_points(ppo_loss_x, ppo_losses, ppo_box, ppo_x_range, ppo_y_range)
    total_points = scale_points(dpo_epochs, total_loss, dpo_box, dpo_x_range, dpo_y_range)
    dpo_points = scale_points(dpo_epochs, dpo_loss, dpo_box, dpo_x_range, dpo_y_range)
    sft_points = scale_points(dpo_epochs, sft_loss, dpo_box, dpo_x_range, dpo_y_range)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="610" viewBox="0 0 1240 610">
  <style>
    .bg {{ fill: #fbfaf7; }}
    .title {{ font: 700 26px Arial, sans-serif; fill: #17202a; }}
    .subtitle {{ font: 15px Arial, sans-serif; fill: #58606a; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #17202a; }}
    .axis {{ stroke: #303742; stroke-width: 1.4; }}
    .grid {{ stroke: #d9ddd8; stroke-width: 1; }}
    .tick {{ font: 12px Arial, sans-serif; fill: #58606a; }}
    .label {{ font: 13px Arial, sans-serif; fill: #303742; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #303742; }}
    .note {{ font: 14px Arial, sans-serif; fill: #303742; }}
    .metric {{ font: 700 19px Arial, sans-serif; fill: #17202a; }}
    .card {{ fill: #ffffff; stroke: #d8ddd6; stroke-width: 1.2; rx: 8; }}
  </style>
  <rect class="bg" x="0" y="0" width="1240" height="610"/>
  <text class="title" x="52" y="48">Retrieval RL Training Convergence</text>
  <text class="subtitle" x="52" y="76">PPO reward/loss trend and DPO preference-training loss over 120 epochs</text>

  <rect class="card" x="48" y="92" width="552" height="332"/>
  <text class="panel-title" x="92" y="134">PPO on-policy signal</text>
  {axis_labels(ppo_box, ppo_x_range, ppo_y_range)}
  <line x1="{ppo_box[0]}" y1="{ppo_box[1] + ppo_box[3]}" x2="{ppo_box[0] + ppo_box[2]}" y2="{ppo_box[1] + ppo_box[3]}" class="axis"/>
  <line x1="{ppo_box[0]}" y1="{ppo_box[1]}" x2="{ppo_box[0]}" y2="{ppo_box[1] + ppo_box[3]}" class="axis"/>
  <polyline points="{ppo_reward_points}" fill="none" stroke="#0f8b8d" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{ppo_loss_points}" fill="none" stroke="#e76f51" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="327" y="404" text-anchor="middle" class="label">Episode</text>
  {legend([("#0f8b8d", "Reward MA"), ("#e76f51", "Loss MA")], 300, 132)}

  <rect class="card" x="646" y="92" width="552" height="332"/>
  <text class="panel-title" x="690" y="134">DPO loss convergence</text>
  {axis_labels(dpo_box, dpo_x_range, dpo_y_range)}
  <line x1="{dpo_box[0]}" y1="{dpo_box[1] + dpo_box[3]}" x2="{dpo_box[0] + dpo_box[2]}" y2="{dpo_box[1] + dpo_box[3]}" class="axis"/>
  <line x1="{dpo_box[0]}" y1="{dpo_box[1]}" x2="{dpo_box[0]}" y2="{dpo_box[1] + dpo_box[3]}" class="axis"/>
  <polyline points="{total_points}" fill="none" stroke="#2b6cb0" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{dpo_points}" fill="none" stroke="#6f42c1" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{sft_points}" fill="none" stroke="#d99021" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="925" y="404" text-anchor="middle" class="label">Epoch</text>
  {legend([("#2b6cb0", "Total loss"), ("#6f42c1", "DPO loss"), ("#d99021", "SFT loss")], 804, 132)}

  <rect class="card" x="48" y="454" width="1150" height="108"/>
  <text class="note" x="78" y="492">DPO total loss:</text>
  <text class="metric" x="202" y="492">{total_loss[0]:.4f} -> {total_loss[-1]:.4f}</text>
  <text class="note" x="420" y="492">DPO preference loss:</text>
  <text class="metric" x="588" y="492">{dpo_loss[0]:.4f} -> {dpo_loss[-1]:.4f}</text>
  <text class="note" x="820" y="492">SFT auxiliary loss:</text>
  <text class="metric" x="970" y="492">{sft_loss[0]:.4f} -> {sft_loss[-1]:.4f}</text>
  <text class="note" x="78" y="532">Interpretation: the DPO objective converges steadily; PPO is noisier because it samples actions online, so the moving average is the useful trend.</text>
</svg>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
