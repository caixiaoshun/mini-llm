#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据单个 TensorBoard 事件文件（events.out.tfevents.*），
估算训练开始时间、结束时间和总时长。

用法：
    python calc_tb_duration.py /path/to/events.out.tfevents.xxx
"""

import argparse
from datetime import datetime

from tensorboard.backend.event_processing import event_accumulator


def format_duration(seconds: float) -> str:
    """把秒数格式化成  X天X小时X分钟X秒  的字符串。"""
    total = int(seconds)
    days = total // (24 * 3600)
    total %= 24 * 3600
    hours = total // 3600
    total %= 3600
    minutes = total // 60
    secs = total % 60

    parts = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分钟")
    # 如果前面都为 0，就至少显示秒
    if secs or not parts:
        parts.append(f"{secs}秒")

    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="从 TensorBoard 事件文件中估算训练时长"
    )
    parser.add_argument(
        "event_file",
        type=str,
        help="TensorBoard 事件文件路径（例如 events.out.tfevents.xxx）",
    )
    args = parser.parse_args()

    event_file = args.event_file

    # 读取事件文件
    print(f"正在解析事件文件: {event_file}")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()
    scalar_tags = tags.get("scalars", [])

    if not scalar_tags:
        print("❌ 日志中没有找到 scalar 数据（比如 loss/accuracy 等）。")
        print("   当前脚本是基于 scalar 事件的时间戳来估算时长的。")
        return

    start_time = None
    end_time = None

    # 遍历所有 scalar 事件，收集最早和最晚的 wall_time
    for tag in scalar_tags:
        events = ea.Scalars(tag)  # 每个元素有 e.wall_time, e.step, e.value
        for e in events:
            t = e.wall_time
            if start_time is None or t < start_time:
                start_time = t
            if end_time is None or t > end_time:
                end_time = t

    if start_time is None or end_time is None:
        print("❌ 没有从事件文件中解析到有效时间戳。")
        return

    duration = end_time - start_time
    if duration < 0:
        print("❌ 解析结果异常：结束时间早于开始时间。")
        return

    start_dt = datetime.fromtimestamp(start_time)
    end_dt = datetime.fromtimestamp(end_time)
    duration_str = format_duration(duration)

    print("✅ 解析完成：")
    print(f"  开始时间: {start_dt}")
    print(f"  结束时间: {end_dt}")
    print(f"  训练大致持续: {duration_str} （约 {duration:.1f} 秒）")


if __name__ == "__main__":
    main()
