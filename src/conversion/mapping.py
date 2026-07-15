"""Map schema-neutral source frames into strict LeRobot frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ConversionError
from ..sources.types import SourceEpisode, SourceFrame
from .plan import ConversionPlan

__all__ = ["map_frame", "resolve_task"]


def resolve_task(plan: ConversionPlan, episode: SourceEpisode) -> str:
    """Resolve natural-language task text from a static value or episode metadata."""

    if plan.task.kind == "static":
        return plan.task.value
    value = episode.metadata.get(plan.task.value)
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(
            "Episode metadata does not provide a valid task",
            context={
                "adapter": plan.adapter,
                "episode": episode.episode_id,
                "selector": plan.task.value,
            },
        )
    return value


def map_frame(plan: ConversionPlan, episode: SourceEpisode, frame: SourceFrame) -> dict[str, Any]:
    """Apply only declared mechanical mappings and add the required LeRobot ``task``."""

    result: dict[str, Any] = {}
    for target, mapping in plan.mappings.items():
        missing = [selector for selector in mapping.sources if selector not in frame.fields]
        if missing:
            raise ConversionError(
                "Source frame is missing a field selected by the plan",
                context={
                    "adapter": plan.adapter,
                    "episode": episode.episode_id,
                    "frame": frame.index,
                    "selector": missing,
                    "target": target,
                },
            )
        try:
            if mapping.operation == "direct":
                value: Any = frame.fields[mapping.sources[0]]
            else:
                value = np.concatenate(
                    [np.asarray(frame.fields[selector]).reshape(-1) for selector in mapping.sources]
                )
            if isinstance(value, np.generic):
                value = np.asarray(value)
            if mapping.cast is not None:
                value = np.asarray(value).astype(mapping.cast, copy=False)

            spec = plan.features[target]
            if spec.dtype in {"image", "video"}:
                if not isinstance(value, np.ndarray) or value.ndim != 3:
                    raise ConversionError(
                        "Image target must be a three-dimensional NumPy array",
                        context={
                            "target": target,
                            "expected_shape": spec.shape,
                            "actual_type": type(value).__name__,
                        },
                    )
                expected = tuple(spec.shape)
                alternative: tuple[int, ...] | None = None
                if expected[-1] in {1, 3, 4}:
                    alternative = (expected[-1], expected[0], expected[1])
                elif expected[0] in {1, 3, 4}:
                    alternative = (expected[1], expected[2], expected[0])
                if value.shape != expected and value.shape != alternative:
                    raise ConversionError(
                        "Image shape is incompatible with the target schema",
                        context={
                            "target": target,
                            "expected": expected,
                            "alternative": alternative,
                            "actual": value.shape,
                        },
                    )
            elif spec.dtype in {"string", "language"}:
                if not isinstance(value, str):
                    raise ConversionError(
                        "String target must be a str value",
                        context={"target": target, "actual_type": type(value).__name__},
                    )
            else:
                if not isinstance(value, np.ndarray):
                    raise ConversionError(
                        "Numeric target must be a NumPy array",
                        context={"target": target, "actual_type": type(value).__name__},
                    )
                if value.dtype != np.dtype(spec.dtype) or value.shape != spec.shape:
                    raise ConversionError(
                        "Numeric target dtype or shape does not match the plan",
                        context={
                            "target": target,
                            "expected_dtype": spec.dtype,
                            "actual_dtype": str(value.dtype),
                            "expected_shape": spec.shape,
                            "actual_shape": value.shape,
                        },
                    )
            result[target] = value
        except ConversionError as exc:
            context = {
                "adapter": plan.adapter,
                "episode": episode.episode_id,
                "frame": frame.index,
                "selector": mapping.sources,
                "target": target,
                **exc.context,
            }
            raise ConversionError(exc.message, context=context) from exc
        except Exception as exc:
            raise ConversionError(
                "Field mapping failed",
                context={
                    "adapter": plan.adapter,
                    "episode": episode.episode_id,
                    "frame": frame.index,
                    "selector": mapping.sources,
                    "target": target,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

    result["task"] = resolve_task(plan, episode)
    return result
