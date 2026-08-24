from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import median
import shlex

@dataclass(frozen=True, slots=True)
class Sample:
    model_path: str
    threads: int
    context: int
    generated_tokens: int
    seconds: float
    peak_rss_bytes: int
    batch: int = 512
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path cannot be empty")
        for name in ("threads", "context", "generated_tokens", "peak_rss_bytes", "batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or self.seconds <= 0:
            raise ValueError("seconds must be positive")

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.model_path, self.threads, self.context, self.batch)

@dataclass(frozen=True, slots=True)
class Profile:
    model_path: str
    threads: int
    context: int
    batch: int
    samples: int
    median_tokens_per_second: float
    minimum_tokens_per_second: float
    peak_rss_bytes: int
    run_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Selection:
    selected: Profile | None
    eligible: tuple[Profile, ...]
    rejected: tuple[dict[str, object], ...]
    command: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": asdict(self.selected) if self.selected else None,
            "eligible": [asdict(item) for item in self.eligible],
            "rejected": list(self.rejected),
            "command": self.command,
        }

def aggregate(samples: list[Sample]) -> tuple[Profile, ...]:
    groups: dict[tuple[str, int, int, int], list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.key, []).append(sample)
    profiles = []
    for key, runs in groups.items():
        speeds = [run.tokens_per_second for run in runs]
        profiles.append(Profile(*key, len(runs), median(speeds), min(speeds), max(run.peak_rss_bytes for run in runs), tuple(run.run_id for run in runs if run.run_id)))
    return tuple(sorted(profiles, key=lambda item: (item.model_path, item.threads, item.context, item.batch)))

def select_profile(samples: list[Sample], *, memory_limit_bytes: int, minimum_tps: float = 0.0, executable: str = "llama-server") -> Selection:
    if memory_limit_bytes <= 0 or minimum_tps < 0:
        raise ValueError("memory_limit_bytes must be positive and minimum_tps non-negative")
    eligible: list[Profile] = []
    rejected: list[dict[str, object]] = []
    for profile in aggregate(samples):
        reasons = []
        if profile.peak_rss_bytes > memory_limit_bytes:
            reasons.append(f"peak RSS exceeds limit by {profile.peak_rss_bytes - memory_limit_bytes} bytes")
        if profile.minimum_tokens_per_second < minimum_tps:
            reasons.append(f"minimum measured throughput is below {minimum_tps:g} tok/s")
        if reasons:
            rejected.append({"profile": asdict(profile), "reasons": reasons})
        else:
            eligible.append(profile)
    selected = max(eligible, key=lambda item: (item.median_tokens_per_second, -item.peak_rss_bytes, -item.threads)) if eligible else None
    command = None
    if selected:
        parts = [executable, "-m", selected.model_path, "-t", str(selected.threads), "-c", str(selected.context), "-b", str(selected.batch)]
        command = shlex.join(parts)
    return Selection(selected, tuple(eligible), tuple(rejected), command)
