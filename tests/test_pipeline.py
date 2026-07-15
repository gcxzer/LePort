"""End-to-end tests for the LeRobot writer, atomic commits, and real 0.6.x reloads."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from leport.api import convert, create_plan, validate
from leport.conversion.pipeline import convert_dataset
from leport.conversion.plan import ConversionPlan, FeatureMapping, FeatureSpec, TargetConfig, TaskProvider
from leport.errors import ConversionError, TargetValidationError
from leport.sources.base import SOURCE_ADAPTER_API_VERSION
from leport.sources.registry import AdapterRegistry
from leport.sources.types import (
    DatasetInspection,
    EpisodeSelection,
    FieldInspection,
    ProbeResult,
    SourceEpisode,
    SourceFrame,
)
from leport.targets.lerobot import validate_lerobot_dataset


def test_numeric_end_to_end_uses_official_lerobot_api(numeric_plan: ConversionPlan) -> None:
    result = convert(numeric_plan)
    assert result.target == numeric_plan.target.root
    assert result.validation.total_episodes == 3
    assert result.validation.total_frames == 9
    assert result.validation.episode_lengths == (3, 4, 2)
    assert result.validation.tasks == ("lift the cube",)

    # Reloading through the installed LeRobot runtime verifies values, indexes, and fixed-FPS timestamps.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=numeric_plan.target.repo_id, root=numeric_plan.target.root)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["action"].numpy(), np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(
        sample["observation.state"].numpy(),
        np.asarray([0, 1, 2, 0, 1], dtype=np.float32),
    )
    assert sample["frame_index"].item() == 0
    assert sample["timestamp"].item() == pytest.approx(0.0)
    assert dataset[1]["timestamp"].item() == pytest.approx(1 / 20)


def test_two_camera_video_end_to_end(robomimic_file: Path, tmp_path: Path) -> None:
    plan = create_plan(
        robomimic_file,
        target_root=tmp_path / "lerobot-video",
        repo_id="tests/robomimic-video",
        fps=20,
        task="lift the cube",
        action_source="actions",
        action_dtype="float32",
        image_sources={
            "obs/agentview_image": "observation.images.agentview",
            "obs/robot0_eye_in_hand_image": "observation.images.wrist",
        },
        adapter="robomimic",
        use_videos=True,
    )
    result = convert(plan)
    assert set(result.validation.decoded_visual_features) == {
        "observation.images.agentview",
        "observation.images.wrist",
    }
    assert result.validation.features["observation.images.agentview"]["dtype"] == "video"
    assert result.validation.episode_lengths == (3, 4, 2)


def test_non_empty_target_is_never_overwritten(numeric_plan: ConversionPlan) -> None:
    numeric_plan.target.root.mkdir()
    marker = numeric_plan.target.root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ConversionError, match="not empty"):
        convert(numeric_plan)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_existing_file_target_is_never_overwritten(numeric_plan: ConversionPlan) -> None:
    numeric_plan.target.root.write_text("keep", encoding="utf-8")
    with pytest.raises(ConversionError, match="not a directory"):
        convert(numeric_plan)
    assert numeric_plan.target.root.read_text(encoding="utf-8") == "keep"


class LateFailureAdapter:
    """Expose a valid preflight frame and a missing second-frame field to test failure cleanup."""

    name: ClassVar[str] = "late-failure"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = None

    def probe(self, source: Path) -> ProbeResult:
        return ProbeResult(self.name, 100, "test")

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        del selection
        return DatasetInspection(
            self.name,
            source,
            ("episode",),
            {"episode": 2},
            (FieldInspection("actions", ("float32",), ((1,),), {"episode": 2}),),
        )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        del source, selection, selectors
        yield SourceEpisode(
            "episode",
            2,
            (
                SourceFrame(0, {"actions": np.zeros(1, dtype=np.float32)}),
                SourceFrame(1, {}),
            ),
        )


class SuccessfulNonHdf5Adapter(LateFailureAdapter):
    """In-memory adapter proving that the generic pipeline does not depend on HDF5."""

    name: ClassVar[str] = "successful-non-hdf5"

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Yield two numeric frames; this synthetic format does not access source or selectors."""

        del source, selection, selectors
        yield SourceEpisode(
            "episode",
            2,
            (
                SourceFrame(0, {"actions": np.asarray([1.0], dtype=np.float32)}),
                SourceFrame(1, {"actions": np.asarray([2.0], dtype=np.float32)}),
            ),
        )


def test_non_hdf5_adapter_reuses_full_conversion_pipeline(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("successful-non-hdf5", SuccessfulNonHdf5Adapter)
    plan = ConversionPlan(
        adapter="successful-non-hdf5",
        source=tmp_path / "synthetic-source",
        selection=EpisodeSelection(),
        target=TargetConfig("tests/successful-non-hdf5", tmp_path / "generic-target", use_videos=False),
        fps=10,
        task=TaskProvider("static", "move"),
        features={"action": FeatureSpec("float32", (1,))},
        mappings={"action": FeatureMapping(("actions",))},
    )
    result = convert_dataset(plan, registry=registry)
    assert result.validation.total_episodes == 1
    assert result.validation.total_frames == 2
    assert result.validation.episode_lengths == (2,)


def test_late_mapping_failure_cleans_temporary_output(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("late-failure", LateFailureAdapter)
    target = tmp_path / "target"
    plan = ConversionPlan(
        adapter="late-failure",
        source=tmp_path / "source",
        selection=EpisodeSelection(),
        target=TargetConfig("tests/late-failure", target, use_videos=False),
        fps=10,
        task=TaskProvider("static", "task"),
        features={"action": FeatureSpec("float32", (1,))},
        mappings={"action": FeatureMapping(("actions",))},
    )
    with pytest.raises(ConversionError) as error:
        convert_dataset(plan, registry=registry)
    assert error.value.context["episode"] == "episode"
    assert error.value.context["frame"] == 1
    assert not target.exists()
    assert not list(tmp_path.glob(".target.leport-*"))


@pytest.mark.parametrize("failure_stage", ["add", "finalize", "validate"])
def test_writer_and_validation_failures_do_not_commit_target(
    numeric_plan: ConversionPlan,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    aborted = {"value": False}

    class ControlledWriter:
        def __init__(self, plan: ConversionPlan, root: Path) -> None:
            del plan, root

        def add_frame(self, frame: dict[str, Any], *, episode_id: str, frame_index: int) -> None:
            del frame, episode_id, frame_index
            if failure_stage == "add":
                raise ConversionError("controlled add failure")

        def save_episode(self, *, episode_id: str) -> None:
            del episode_id

        def finalize(self) -> None:
            if failure_stage == "finalize":
                raise ConversionError("controlled finalize failure")

        def abort(self) -> None:
            aborted["value"] = True

    monkeypatch.setattr("leport.conversion.pipeline.LeRobotDatasetWriter", ControlledWriter)
    if failure_stage == "validate":
        monkeypatch.setattr(
            "leport.conversion.pipeline.validate_lerobot_dataset",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                TargetValidationError("controlled readback failure")
            ),
        )

    with pytest.raises((ConversionError, TargetValidationError)):
        convert_dataset(numeric_plan)
    assert aborted["value"]
    assert not numeric_plan.target.root.exists()
    assert not list(numeric_plan.target.root.parent.glob(f".{numeric_plan.target.root.name}.leport-*"))


def test_validate_can_compare_existing_target_with_source(numeric_plan: ConversionPlan) -> None:
    convert(numeric_plan)
    report = validate(numeric_plan.target.root, plan=numeric_plan)
    assert report.total_frames == 9
    assert report.episode_lengths == (3, 4, 2)


def test_validate_can_read_target_without_plan_or_repo_id(numeric_plan: ConversionPlan) -> None:
    convert(numeric_plan)
    # Standalone validation uses target metadata only; the local placeholder repo_id never accesses Hub.
    report = validate(numeric_plan.target.root)
    assert report.total_episodes == 3
    assert report.total_frames == 9
    assert report.repo_id == "local/lerobot-numeric"


def test_episode_length_mismatch_reports_source_episode_id(numeric_plan: ConversionPlan) -> None:
    convert(numeric_plan)
    with pytest.raises(TargetValidationError, match="episode lengths") as error:
        validate_lerobot_dataset(
            numeric_plan.target.root,
            repo_id=numeric_plan.target.repo_id,
            expected_episode_ids=("demo_0", "demo_2", "demo_10"),
            expected_episode_lengths=(3, 5, 2),
        )
    assert error.value.context["mismatches"] == {"demo_2": {"expected": 5, "actual": 4}}
