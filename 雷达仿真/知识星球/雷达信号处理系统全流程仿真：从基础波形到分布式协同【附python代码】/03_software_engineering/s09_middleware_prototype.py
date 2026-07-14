"""s09：消息中间件原型 — 模拟 ZeroMQ PUB/SUB 模式。

设计模式：
  使用 Python 标准库 queue + threading 模拟 ZeroMQ 的 PUB/SUB 消息分发模式，
  实现雷达节点间的数据分发原型。不需要安装 zmq 库。

数据流拓扑：
  RadarDataPublisher (radar/iq)
       │
       ├── PulseCompressSubscriber ──→ (radar/pc)
       │        │
       │        ├── DetectorSubscriber ──→ (radar/detections)
       │        │
       │        └── MonitorSubscriber (radar/pc)
       │
       └── MonitorSubscriber (radar/iq, radar/detections)

核心组件：
  1. MessageBus    — 基于 queue.Queue 的发布/订阅消息总线
  2. Publisher      — 雷达 IQ 数据源节点
  3. Subscriber     — 信号处理链节点（脉冲压缩、检测）
  4. Monitor        — 全链路监控节点

验证项：
  1. 数据完整性：发布的数据能被订阅者完整接收
  2. 延迟测量：发布-接收延迟 < 100 ms（本地 queue）
  3. 多订阅者：同一主题的多个订阅者都能收到数据
  4. 处理链：IQ → 脉冲压缩 → 检测完整数据流
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from lib.radar_params import RadarParams, SPEED_OF_LIGHT
from lib.signal_utils import generate_lfm, matched_filter, power_to_db
from lib.validation import verify, print_validation, ValidationResult

# ─────────────────────────────────────────────
# 1. MessageBus — 基于 queue 的发布/订阅消息总线
# ─────────────────────────────────────────────


class MessageBus:
    """模拟 ZeroMQ PUB/SUB 模式的消息总线。

    内部用 dict[str, queue.Queue] 管理主题到队列的映射。
    每个订阅者获得独立的队列（fan-out 模式），
    发布的消息会被复制到该主题的所有订阅者队列中。

    线程安全说明：
      - publish() 由发布者线程调用
      - subscribe() 返回的 Queue 由订阅者线程读取
      - Queue 本身是线程安全的，无需额外锁
    """

    def __init__(self) -> None:
        # 主题 → 订阅者队列列表
        self._topics: dict[str, list[queue.Queue]] = {}
        self._lock = threading.Lock()

    def create_topic(self, name: str) -> None:
        """创建主题（幂等操作，已存在则忽略）。"""
        with self._lock:
            if name not in self._topics:
                self._topics[name] = []

    def publish(self, topic: str, data: Any) -> int:
        """发布数据到指定主题。

        将数据复制到该主题的所有订阅者队列。
        返回接收到消息的订阅者数量。

        Args:
            topic: 主题名称
            data:  任意数据（雷达场景中通常是 numpy 数组或 dataclass）

        Returns:
            接收消息的订阅者队列数
        """
        with self._lock:
            queues = self._topics.get(topic, [])
            count = 0
            for q in queues:
                try:
                    q.put_nowait(data)
                    count += 1
                except queue.Full:
                    # 队列满时丢弃（PUB/SUB 的 fire-and-forget 语义）
                    pass
            return count

    def subscribe(self, topic: str, maxsize: int = 1024) -> queue.Queue:
        """订阅主题，返回数据队列。

        调用者从返回的 Queue 中 get() 获取数据。
        Queue 为空时 get() 会阻塞，支持迭代式消费。

        Args:
            topic:   主题名称
            maxsize: 队列最大长度（防止慢消费者堆积）

        Returns:
            订阅者专属的 queue.Queue
        """
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []
            self._topics[topic].append(q)
        return q

    def topic_names(self) -> list[str]:
        """返回所有已创建的主题名称。"""
        with self._lock:
            return list(self._topics.keys())

    def subscriber_count(self, topic: str) -> int:
        """返回指定主题的订阅者数量。"""
        with self._lock:
            return len(self._topics.get(topic, []))


# ─────────────────────────────────────────────
# 2. 消息封装
# ─────────────────────────────────────────────


@dataclass
class RadarMessage:
    """雷达数据消息封装。

    在真实系统中对应 ZeroMQ 的 multipart message：
      frame 0: 元数据（JSON/dict）
      frame 1: 载荷（numpy bytes）

    Attributes:
        timestamp_ns: 发布时的 monotonic 时间戳 (ns)
        block_index:  数据块序号（用于完整性校验）
        payload:      载荷数据（numpy 数组或检测结果列表）
    """
    timestamp_ns: int
    block_index: int
    payload: Any


@dataclass
class Detection:
    """单次检测结果。

    Attributes:
        range_m:     检测目标距离 (m)
        snr_db:      检测信噪比 (dB)
        block_index: 来源数据块序号
    """
    range_m: float
    snr_db: float
    block_index: int


# ─────────────────────────────────────────────
# 3. Publisher — 雷达 IQ 数据源
# ─────────────────────────────────────────────


class RadarDataPublisher:
    """模拟雷达数据源节点。

    生成 LFM 脉冲回波的 IQ 数据块，每个块包含：
      - 发射信号模板（用于匹配滤波）
      - 接收信号（含目标回波 + 噪声）

    以固定间隔发布到 "radar/iq" 主题，模拟实时数据流。
    """

    def __init__(
        self,
        bus: MessageBus,
        params: RadarParams,
        num_blocks: int = 50,
        interval_s: float = 0.01,
    ) -> None:
        """
        Args:
            bus:        消息总线
            params:     雷达参数
            num_blocks: 发送的数据块总数
            interval_s: 每块之间的发送间隔 (s)
        """
        self.bus = bus
        self.params = params
        self.num_blocks = num_blocks
        self.interval_s = interval_s
        self.rng = np.random.default_rng(seed=42)

        # 预计算 LFM 模板（所有块共享同一模板）
        sample_rate_hz = 2 * params.bandwidth_hz  # Nyquist: fs > B
        self.template = generate_lfm(
            bandwidth_hz=params.bandwidth_hz,
            pulse_width_s=params.pulse_width_s,
            sample_rate_hz=sample_rate_hz,
        )
        self.sample_rate_hz = sample_rate_hz

    def generate_iq_block(self, block_index: int) -> RadarMessage:
        """生成一个 IQ 数据块。

        模拟接收信号 = 目标回波 + 热噪声。
        目标回波通过 LFM 信号的时延来模拟。
        """
        params = self.params
        n_samples = len(self.template)

        # 目标回波：LFM 信号 + 时延（用距离换算采样点偏移）
        delay_s = 2 * params.target_range_m / SPEED_OF_LIGHT
        delay_samples = int(delay_s * self.sample_rate_hz)

        # 接收信号：零填充 + 噪声
        rx_signal = np.zeros(n_samples + delay_samples, dtype=np.complex128)

        # 目标回波嵌入（简化：信号从 delay_samples 位置开始）
        signal_power = 1e-6  # 归一化信号功率 (W)
        rx_signal[delay_samples: delay_samples + n_samples] += (
            np.sqrt(signal_power) * self.template
        )

        # 热噪声
        noise_power = params.noise_power_w
        noise = self.rng.normal(0, np.sqrt(noise_power / 2), len(rx_signal)) + \
                1j * self.rng.normal(0, np.sqrt(noise_power / 2), len(rx_signal))
        rx_signal += noise

        # 封装为消息
        payload = {
            "rx_signal": rx_signal,
            "template": self.template,
            "sample_rate_hz": self.sample_rate_hz,
        }
        return RadarMessage(
            timestamp_ns=time.monotonic_ns(),
            block_index=block_index,
            payload=payload,
        )

    def run(self) -> list[int]:
        """发布所有数据块。返回已发送的块序号列表。"""
        sent_indices = []
        for i in range(self.num_blocks):
            msg = self.generate_iq_block(i)
            self.bus.publish("radar/iq", msg)
            sent_indices.append(i)
            time.sleep(self.interval_s)
        # 发送结束信号
        self.bus.publish("radar/iq", None)
        return sent_indices


# ─────────────────────────────────────────────
# 4. Subscriber — 脉冲压缩节点
# ─────────────────────────────────────────────


class PulseCompressSubscriber:
    """脉冲压缩处理节点。

    订阅 "radar/iq"，对每个 IQ 数据块做匹配滤波（脉冲压缩），
    将压缩结果发布到 "radar/pc"。

    匹配滤波原理：
      频域实现 Y(f) = X(f) × H*(f)
      其中 X(f) 是接收信号频谱，H(f) 是模板频谱。
      压缩后在目标位置产生峰值，获得 T×B 增益。
    """

    def __init__(self, bus: MessageBus, topic_in: str = "radar/iq",
                 topic_out: str = "radar/pc") -> None:
        self.bus = bus
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.processed_count = 0

    def run(self) -> None:
        """消费 IQ 数据，执行脉冲压缩，发布结果。"""
        q = self.bus.subscribe(self.topic_in)

        while True:
            msg: Optional[RadarMessage] = q.get()
            if msg is None:
                # 传播结束信号
                self.bus.publish(self.topic_out, None)
                break

            payload = msg.payload
            rx_signal = payload["rx_signal"]
            template = payload["template"]

            # 匹配滤波（脉冲压缩）
            compressed = matched_filter(rx_signal, template)

            # 发布压缩结果
            pc_msg = RadarMessage(
                timestamp_ns=time.monotonic_ns(),
                block_index=msg.block_index,
                payload={
                    "compressed": compressed,
                    "original_timestamp_ns": msg.timestamp_ns,
                },
            )
            self.bus.publish(self.topic_out, pc_msg)
            self.processed_count += 1


# ─────────────────────────────────────────────
# 5. Subscriber — 检测节点
# ─────────────────────────────────────────────


class DetectorSubscriber:
    """CFAR 检测节点（简化版）。

    订阅 "radar/pc"，对压缩后的信号做简单的阈值检测。
    将检测结果发布到 "radar/detections"。

    简化说明：使用固定阈值检测（真实系统用 CFAR），
    阈值基于噪声基底的统计特性设定。
    """

    def __init__(
        self,
        bus: MessageBus,
        threshold_db: float = 13.0,
        topic_in: str = "radar/pc",
        topic_out: str = "radar/detections",
    ) -> None:
        self.bus = bus
        self.threshold_db = threshold_db
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.processed_count = 0

    def run(self) -> None:
        """消费脉冲压缩结果，执行检测，发布检测报告。"""
        q = self.bus.subscribe(self.topic_in)

        while True:
            msg: Optional[RadarMessage] = q.get()
            if msg is None:
                self.bus.publish(self.topic_out, None)
                break

            compressed = msg.payload["compressed"]
            original_ts = msg.payload["original_timestamp_ns"]

            # 计算压缩信号功率（dB）
            magnitude = np.abs(compressed)
            peak_power = np.max(magnitude ** 2)
            noise_floor = np.median(magnitude ** 2)
            snr_linear = peak_power / max(noise_floor, 1e-40)
            snr_db = power_to_db(snr_linear)

            # 简单阈值检测
            detections = []
            if snr_db > self.threshold_db:
                # 峰值位置 → 距离（简化：假设已知采样率）
                peak_idx = np.argmax(magnitude)
                # 这里不做精确的距离换算，用峰值索引作为标识
                detections.append(Detection(
                    range_m=float(peak_idx),  # 简化：用索引代替真实距离
                    snr_db=float(snr_db),
                    block_index=msg.block_index,
                ))

            # 发布检测结果
            det_msg = RadarMessage(
                timestamp_ns=time.monotonic_ns(),
                block_index=msg.block_index,
                payload={
                    "detections": detections,
                    "original_timestamp_ns": original_ts,
                },
            )
            self.bus.publish(self.topic_out, det_msg)
            self.processed_count += 1


# ─────────────────────────────────────────────
# 6. Monitor — 全链路监控
# ─────────────────────────────────────────────


@dataclass
class MonitorRecord:
    """单条监控记录。"""
    topic: str
    block_index: int
    publish_timestamp_ns: int
    receive_timestamp_ns: int
    latency_ms: float


class MonitorSubscriber:
    """监控节点：订阅指定主题，记录延迟和吞吐量。

    用于验证中间件原型的性能指标：
      - 端到端延迟
      - 消息吞吐量
      - 数据完整性（是否丢块）
    """

    def __init__(self, bus: MessageBus, topics: list[str]) -> None:
        self.bus = bus
        self.topics = topics
        self.records: list[MonitorRecord] = []
        self._threads: list[threading.Thread] = []

    def _consume_topic(self, topic: str) -> None:
        """消费单个主题的消息并记录延迟。"""
        q = self.bus.subscribe(topic)
        while True:
            msg: Optional[RadarMessage] = q.get()
            if msg is None:
                break
            now_ns = time.monotonic_ns()
            latency_ms = (now_ns - msg.timestamp_ns) / 1e6
            self.records.append(MonitorRecord(
                topic=topic,
                block_index=msg.block_index,
                publish_timestamp_ns=msg.timestamp_ns,
                receive_timestamp_ns=now_ns,
                latency_ms=latency_ms,
            ))

    def start(self) -> None:
        """启动监控线程（每个主题一个线程）。"""
        for topic in self.topics:
            t = threading.Thread(
                target=self._consume_topic, args=(topic,), daemon=True
            )
            self._threads.append(t)
            t.start()

    def wait_and_collect(self) -> None:
        """等待所有监控线程结束。"""
        for t in self._threads:
            t.join(timeout=30)

    def get_records_by_topic(self, topic: str) -> list[MonitorRecord]:
        """获取指定主题的所有监控记录。"""
        return [r for r in self.records if r.topic == topic]


# ─────────────────────────────────────────────
# 7. 绘制数据流拓扑图
# ─────────────────────────────────────────────


def plot_dataflow_topology(output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")) -> None:
    """绘制数据流拓扑图，保存为 PNG。"""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import FancyArrowPatch

    rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("s09 消息中间件原型 — 数据流拓扑", fontsize=14, fontweight="bold")

    # 节点位置
    nodes = {
        "RadarDataPublisher": (0.5, 3.5),
        "radar/iq": (2, 3.5),
        "PulseCompressSubscriber": (2, 2.2),
        "radar/pc": (3.5, 2.2),
        "DetectorSubscriber": (3.5, 1.0),
        "radar/detections": (3.5, 0.0),
        "MonitorSubscriber": (0.5, 0.5),
    }

    # 绘制节点
    for name, (x, y) in nodes.items():
        color = "#4CAF50" if "Publisher" in name or "Subscriber" in name else "#2196F3"
        if "Monitor" in name:
            color = "#FF9800"
        bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor="black")
        ax.text(x, y, name, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", bbox=bbox)

    # 绘制连线（带箭头）
    edges = [
        ("RadarDataPublisher", "radar/iq"),
        ("radar/iq", "PulseCompressSubscriber"),
        ("PulseCompressSubscriber", "radar/pc"),
        ("radar/pc", "DetectorSubscriber"),
        ("DetectorSubscriber", "radar/detections"),
        ("radar/iq", "MonitorSubscriber"),
        ("radar/pc", "MonitorSubscriber"),
        ("radar/detections", "MonitorSubscriber"),
    ]

    for src, dst in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            color="#555555",
            mutation_scale=15,
            linewidth=1.5,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    # 图例
    legend_items = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
                    markersize=12, label="处理节点 (Publisher/Subscriber)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2196F3",
                    markersize=12, label="主题 (Topic)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
                    markersize=12, label="监控节点 (Monitor)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "s09_middleware.png"), dpi=150, bbox_inches="tight")
    print(f"  拓扑图已保存: {output_dir}/s09_middleware.png")
    plt.close(fig)


# ─────────────────────────────────────────────
# 8. 验证
# ─────────────────────────────────────────────


def validate(
    sent_indices: list[int],
    monitor: MonitorSubscriber,
    pc_subscriber: PulseCompressSubscriber,
    detector_subscriber: DetectorSubscriber,
) -> bool:
    """验证中间件原型的四项指标。"""
    results: list[ValidationResult] = []

    # --- 验证 1：数据完整性 ---
    # 检查 IQ 主题的监控记录是否覆盖所有已发送的块
    iq_records = monitor.get_records_by_topic("radar/iq")
    received_indices = sorted(set(r.block_index for r in iq_records))
    sent_set = set(sent_indices)
    received_set = set(received_indices)
    missing = sent_set - received_set
    loss_rate = len(missing) / len(sent_set) if sent_set else 0.0

    results.append(verify(
        name="数据完整性（IQ 丢失率）",
        theoretical=0.0,
        simulated=loss_rate,
        tolerance=0.0,  # 不允许丢失
        unit="",
    ))

    # --- 验证 2：延迟测量 ---
    # 本地 queue 的发布-接收延迟应 < 100 ms
    iq_latencies = [r.latency_ms for r in iq_records]
    max_latency = max(iq_latencies) if iq_latencies else 999.0
    results.append(verify(
        name="最大延迟 < 100ms",
        theoretical=0.0,
        simulated=max_latency,
        tolerance=100.0,  # 允许 100 ms
        unit="ms",
    ))

    # --- 验证 3：多订阅者 ---
    # "radar/iq" 应有 2 个订阅者（PulseCompressSubscriber + MonitorSubscriber）
    # 但我们通过 Monitor 只能看到到达 Monitor 的消息
    # 用 PulseCompressSubscriber 的处理数来间接验证
    results.append(verify(
        name="脉冲压缩处理数",
        theoretical=float(len(sent_indices)),
        simulated=float(pc_subscriber.processed_count),
        tolerance=0.0,
        unit="块",
    ))

    # --- 验证 4：处理链完整性 ---
    # 检测节点应处理了与发送数相同的块
    results.append(verify(
        name="检测处理数",
        theoretical=float(len(sent_indices)),
        simulated=float(detector_subscriber.processed_count),
        tolerance=0.0,
        unit="块",
    ))

    # --- 验证 5：延迟统计 ---
    pc_records = monitor.get_records_by_topic("radar/pc")
    pc_latencies = [r.latency_ms for r in pc_records]
    avg_latency = np.mean(iq_latencies) if iq_latencies else 0.0
    avg_pc_latency = np.mean(pc_latencies) if pc_latencies else 0.0

    print(f"\n  延迟统计:")
    print(f"    IQ 平均延迟:   {avg_latency:.2f} ms")
    print(f"    PC 平均延迟:   {avg_pc_latency:.2f} ms")
    print(f"    IQ 最大延迟:   {max_latency:.2f} ms")
    print(f"    总消息数:      {len(monitor.records)}")
    print(f"    监控主题:      {monitor.topics}")

    return print_validation("s09 消息中间件原型", results)


# ─────────────────────────────────────────────
# 9. 主函数
# ─────────────────────────────────────────────


def main() -> int:
    """运行 s09 消息中间件原型仿真与验证。"""
    print("=" * 60)
    print("s09：消息中间件原型 — 模拟 ZeroMQ PUB/SUB 模式")
    print("=" * 60)

    # 雷达参数（简化配置，用于快速验证）
    params = RadarParams(
        pt=1e6,
        gain_db=30.0,
        freq_hz=10e9,        # X 波段
        bandwidth_hz=10e6,   # 10 MHz 带宽
        pulse_width_s=10e-6, # 10 μs 脉宽
        prf_hz=1000.0,
        noise_figure_db=3.0,
        target_range_m=30e3,
        target_rcs_m2=5.0,
    )

    num_blocks = 50
    interval_s = 0.005  # 5 ms 间隔

    print(f"\n中间件配置:")
    print(f"  模拟方式: queue + threading（无需 zmq）")
    print(f"  数据块数: {num_blocks}")
    print(f"  发送间隔: {interval_s * 1000:.0f} ms")
    print(f"  预计耗时: {num_blocks * interval_s:.1f} s")

    # --- 初始化消息总线 ---
    bus = MessageBus()
    bus.create_topic("radar/iq")
    bus.create_topic("radar/pc")
    bus.create_topic("radar/detections")

    # --- 初始化节点 ---
    publisher = RadarDataPublisher(
        bus=bus, params=params, num_blocks=num_blocks, interval_s=interval_s,
    )
    pc_subscriber = PulseCompressSubscriber(bus=bus)
    detector_subscriber = DetectorSubscriber(bus=bus, threshold_db=13.0)

    # 监控节点：订阅所有主题
    monitor = MonitorSubscriber(
        bus=bus, topics=["radar/iq", "radar/pc", "radar/detections"],
    )

    print(f"\n启动节点...")
    print(f"  主题列表: {bus.topic_names()}")

    # 启动监控线程
    monitor.start()

    # 启动处理链（每个 subscriber 一个线程）
    pc_thread = threading.Thread(target=pc_subscriber.run, daemon=True)
    det_thread = threading.Thread(target=detector_subscriber.run, daemon=True)
    pc_thread.start()
    det_thread.start()

    # 等待订阅者就绪（短暂等待，确保队列已注册）
    time.sleep(0.05)

    # 启动发布者
    print(f"  发布者开始发送数据...")
    sent_indices = publisher.run()
    print(f"  发布者发送完毕，共 {len(sent_indices)} 块")

    # 等待处理链完成
    pc_thread.join(timeout=30)
    det_thread.join(timeout=30)

    # 等待监控收集完毕
    monitor.wait_and_collect()

    # --- 绘制拓扑图 ---
    print(f"\n绘制数据流拓扑图...")
    plot_dataflow_topology()

    # --- 验证 ---
    print(f"\n运行验证...")
    all_passed = validate(sent_indices, monitor, pc_subscriber, detector_subscriber)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
