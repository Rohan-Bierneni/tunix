# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for trajectory statistics derivation."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer.diagnostics import stats as stats_lib

_AGENT = trajectory_lib.Agent(name="a", version="1.0")


def _trajectory(
    steps: list[trajectory_lib.Step] | None = None,
    **kwargs: Any,
) -> trajectory_lib.Trajectory[trajectory_lib.Step]:
  """Builds a base ATIF trajectory with the given steps and metadata."""
  return trajectory_lib.Trajectory(
      trajectory_id="traj_test",
      agent=_AGENT,
      steps=steps or [],
      **kwargs,
  )


def _step(
    step_id: int,
    source: trajectory_lib.Source,
    **kwargs: Any,
) -> trajectory_lib.Step:
  """Builds a base ATIF step."""
  return trajectory_lib.Step(
      step_id=step_id, source=source, message="m", **kwargs
  )


def _tunix_trajectory(
    trajectory_id: str = "traj_test", **kwargs: Any
) -> trajectory_lib.TunixTrajectory:
  """Builds a Tunix trajectory, which can report `status` and `total_reward`."""
  return trajectory_lib.TunixTrajectory(
      trajectory_id=trajectory_id, agent=_AGENT, **kwargs
  )


class StepCountsTest(absltest.TestCase):

  def test_counts_are_split_by_source(self) -> None:
    trajectory = _trajectory([
        _step(1, trajectory_lib.Source.SYSTEM),
        _step(2, trajectory_lib.Source.USER),
        _step(3, trajectory_lib.Source.AGENT),
        _step(4, trajectory_lib.Source.AGENT),
    ])

    steps = stats_lib.TrajectoryStats.from_trajectory(trajectory).steps

    self.assertEqual(steps, stats_lib.StepCounts(agent=2, user=1, system=1))
    self.assertEqual(steps.total, 4)
    self.assertEqual(steps.env_steps, 2)

  def test_total_is_the_sum_of_the_source_split(self) -> None:
    # The fields partition the steps, so `total` is derived rather than stored
    # and `sum(counts) == counts.total` holds by construction.
    counts = stats_lib.StepCounts(agent=2, user=2, other=1)

    self.assertEqual(counts.total, 5)
    self.assertEqual(sum(counts), counts.total)
    self.assertEqual(
        counts.to_json_dict(),
        {"total": 5, "agent": 2, "user": 2, "system": 0, "other": 1},
    )


class TokenCountsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("no_metrics", None, None),
      (
          "metrics_without_token_fields",
          trajectory_lib.Metrics(cost_usd=0.01),
          None,
      ),
      (
          "partial_token_fields",
          trajectory_lib.Metrics(prompt_tokens=7),
          stats_lib.TokenCounts(prompt=7),
      ),
      (
          "all_token_fields",
          trajectory_lib.Metrics(
              prompt_tokens=10, completion_tokens=2, cached_tokens=1
          ),
          stats_lib.TokenCounts(prompt=10, completion=2, cached=1),
      ),
  )
  def test_from_metrics(
      self,
      metrics: trajectory_lib.Metrics | None,
      expected: stats_lib.TokenCounts | None,
  ) -> None:
    self.assertEqual(stats_lib.TokenCounts.from_metrics(metrics), expected)

  def test_final_metrics_are_authoritative(self) -> None:
    trajectory = _trajectory(
        [
            _step(
                1,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(
                    prompt_tokens=1, completion_tokens=1
                ),
            )
        ],
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=100,
            total_completion_tokens=20,
            total_cached_tokens=50,
        ),
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(
        tokens, stats_lib.TokenCounts(prompt=100, completion=20, cached=50)
    )
    self.assertEqual(tokens.total, 120)

  def test_steps_are_summed_when_final_metrics_are_absent(self) -> None:
    trajectory = _trajectory([
        _step(1, trajectory_lib.Source.USER),
        _step(
            2,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(
                prompt_tokens=10, completion_tokens=2, cached_tokens=1
            ),
        ),
        _step(
            3,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(
                prompt_tokens=20, completion_tokens=3
            ),
        ),
    ])

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(
        tokens, stats_lib.TokenCounts(prompt=30, completion=5, cached=1)
    )

  def test_steps_are_summed_when_final_metrics_carry_no_token_totals(
      self,
  ) -> None:
    trajectory = _trajectory(
        [
            _step(
                1,
                trajectory_lib.Source.AGENT,
                metrics=trajectory_lib.Metrics(prompt_tokens=7),
            )
        ],
        final_metrics=trajectory_lib.FinalMetrics(total_steps=1),
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(tokens, stats_lib.TokenCounts(prompt=7))

  def test_a_measured_zero_is_kept(self) -> None:
    trajectory = _trajectory(
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=0, total_completion_tokens=0
        )
    )

    tokens = stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens

    self.assertEqual(tokens, stats_lib.TokenCounts())

  def test_steps_with_metrics_but_no_token_fields_are_unreported(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            metrics=trajectory_lib.Metrics(cost_usd=0.01),
        )
    ])

    self.assertIsNone(
        stats_lib.TrajectoryStats.from_trajectory(trajectory).tokens
    )


class ToolCallsTest(absltest.TestCase):

  def test_sequence_follows_invocation_order(self) -> None:
    trajectory = _trajectory([
        _step(
            1,
            trajectory_lib.Source.AGENT,
            tool_calls=[
                trajectory_lib.ToolCall(tool_call_id="1", function_name="b"),
                trajectory_lib.ToolCall(tool_call_id="2", function_name="a"),
            ],
        ),
        _step(
            2,
            trajectory_lib.Source.AGENT,
            tool_calls=[
                trajectory_lib.ToolCall(tool_call_id="3", function_name="b")
            ],
        ),
    ])

    names = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).tool_call_names

    self.assertEqual(names, ("b", "a", "b"))


class OutcomeTest(parameterized.TestCase):
  """Tests for `status` and `total_reward`, which are not base ATIF fields."""

  def test_schemas_that_declare_the_fields_report_them(self) -> None:
    trajectory = _tunix_trajectory(total_reward=0.5, status="SUCCEEDED")

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertEqual(stats.status, stats_lib.TrajectoryStatus.SUCCEEDED)
    self.assertEqual(stats.total_reward, 0.5)

  def test_schemas_that_do_not_declare_the_fields_report_neither(self) -> None:
    trajectory = _trajectory(extra={"status": "SUCCEEDED", "reward": 1.0})

    stats = stats_lib.TrajectoryStats.from_trajectory(trajectory)

    self.assertIsNone(stats.status)
    self.assertIsNone(stats.total_reward)

  def test_an_unstated_status_is_unreported(self) -> None:
    stats = stats_lib.TrajectoryStats.from_trajectory(_tunix_trajectory())

    self.assertIsNone(stats.status)

  def test_an_unrecognized_status_is_rejected(self) -> None:
    trajectory = _tunix_trajectory(status="COMPLETED")

    with self.assertRaisesRegex(ValueError, "Unrecognized trajectory status"):
      stats_lib.TrajectoryStats.from_trajectory(trajectory)

  @parameterized.named_parameters(
      ("nan", float("nan")),
      ("inf", float("inf")),
      ("negative_inf", float("-inf")),
  )
  def test_a_non_finite_total_reward_is_unreported(self, value: float) -> None:
    trajectory = _tunix_trajectory(total_reward=value)

    self.assertIsNone(
        stats_lib.TrajectoryStats.from_trajectory(trajectory).total_reward
    )


class RunStatsTest(parameterized.TestCase):

  def test_df_has_one_typed_row_per_trajectory(self) -> None:
    reporting = _tunix_trajectory(
        trajectory_id="traj_1",
        steps=[
            trajectory_lib.TunixAgentStep(
                step_id=0,
                source=trajectory_lib.Source.AGENT,
                message="m",
                tool_calls=[
                    trajectory_lib.ToolCall(tool_call_id="0", function_name="a"),
                    trajectory_lib.ToolCall(tool_call_id="1", function_name="b"),
                ],
            ),
            trajectory_lib.TunixEnvStep(
                step_id=1, source=trajectory_lib.Source.USER, message="m"
            ),
        ],
        status="SUCCEEDED",
        total_reward=0.5,
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=5, total_completion_tokens=2
        ),
    )
    silent = _tunix_trajectory(trajectory_id="traj_2")

    run = stats_lib.RunStats.from_trajectories([reporting, silent])

    self.assertEqual(run.df.columns.tolist(), list(stats_lib.COLUMN_DTYPES))
    self.assertEqual(
        run.df.dtypes.astype(str).to_dict(), dict(stats_lib.COLUMN_DTYPES)
    )
    self.assertEqual(
        run.df.iloc[0].to_dict(),
        stats_lib.TrajectoryStats.from_trajectory(reporting).to_row(),
    )
    self.assertEqual(
        run.df.iloc[0].to_dict(),
        {
            "trajectory_id": "traj_1",
            "status": "SUCCEEDED",
            "total_reward": 0.5,
            "steps_agent": 1,
            "steps_user": 1,
            "steps_system": 0,
            "steps_other": 0,
            "tokens_prompt": 5,
            "tokens_completion": 2,
            "tokens_cached": 0,
            "num_tool_calls": 2,
        },
    )
    # Unreported quantities are nulls, not zeros.
    self.assertEqual(run.df["trajectory_id"].tolist(), ["traj_1", "traj_2"])
    self.assertEqual(
        run.df.iloc[1].isna().to_dict(),
        {
            "trajectory_id": False,
            "status": True,
            "total_reward": True,
            "steps_agent": False,
            "steps_user": False,
            "steps_system": False,
            "steps_other": False,
            "tokens_prompt": True,
            "tokens_completion": True,
            "tokens_cached": True,
            "num_tool_calls": False,
        },
    )

  def test_empty_run_keeps_the_column_schema(self) -> None:
    run = stats_lib.RunStats.from_trajectories([])

    self.assertEmpty(run.df)
    self.assertEqual(
        run.df.dtypes.astype(str).to_dict(), dict(stats_lib.COLUMN_DTYPES)
    )

  def test_totals_sum_over_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _trajectory(
            [_step(i, trajectory_lib.Source.AGENT) for i in range(1, 7)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=520,
                total_completion_tokens=90,
                total_cached_tokens=220,
            ),
        ),
        _trajectory(
            [_step(1, trajectory_lib.Source.USER)]
            + [_step(i, trajectory_lib.Source.AGENT) for i in range(2, 5)],
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=180,
                total_completion_tokens=35,
                total_cached_tokens=0,
            ),
        ),
        _trajectory([_step(1, trajectory_lib.Source.SYSTEM)]),
    ])

    self.assertEqual(run.num_trajectories, 3)
    self.assertEqual(run.steps, stats_lib.StepCounts(agent=9, user=1, system=1))
    self.assertEqual(
        run.tokens,
        stats_lib.TokenCounts(prompt=700, completion=125, cached=220),
    )
    self.assertEqual(run.num_with_tokens, 2)

  def test_mean_reward_averages_only_reporting_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(total_reward=1.0),
        _tunix_trajectory(total_reward=0.0),
        _tunix_trajectory(),
    ])

    self.assertEqual(run.df["total_reward"].dropna().tolist(), [1.0, 0.0])
    self.assertEqual(run.mean_reward, 0.5)
    self.assertEqual(run.num_with_reward, 2)

  @parameterized.named_parameters(
      ("empty_run", 0),
      ("trajectories_reporting_nothing", 2),
  )
  def test_quantities_are_unreported_when_no_trajectory_reports_them(
      self, num_trajectories: int
  ) -> None:
    run = stats_lib.RunStats.from_trajectories(
        [_trajectory() for _ in range(num_trajectories)]
    )

    self.assertEqual(run.num_trajectories, num_trajectories)
    self.assertEqual(run.steps.total, 0)
    self.assertIsNone(run.tokens)
    self.assertEqual(run.num_with_tokens, 0)
    self.assertIsNone(run.mean_reward)
    self.assertEqual(run.num_with_reward, 0)
    self.assertEmpty(run.status_counts)
    self.assertEqual(run.num_with_status, 0)
    self.assertEmpty(run.tool_counts)

  def test_status_counts_tally_only_reporting_trajectories(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(status="TIMEOUT"),
        _tunix_trajectory(status="SUCCEEDED"),
        _tunix_trajectory(),
        _tunix_trajectory(status="SUCCEEDED"),
    ])

    self.assertEqual(
        run.status_counts,
        {
            stats_lib.TrajectoryStatus.SUCCEEDED: 2,
            stats_lib.TrajectoryStatus.TIMEOUT: 1,
        },
    )
    self.assertEqual(
        list(run.status_counts),
        [
            stats_lib.TrajectoryStatus.TIMEOUT,
            stats_lib.TrajectoryStatus.SUCCEEDED,
        ],
    )
    self.assertEqual(run.num_with_status, 3)
    self.assertEqual(
        run.to_json_dict()["status_counts"], {"SUCCEEDED": 2, "TIMEOUT": 1}
    )
    self.assertEqual(
        list(run.to_json_dict()["status_counts"]), ["SUCCEEDED", "TIMEOUT"]
    )

  def test_tool_counts_are_summed_across_the_run(self) -> None:
    def with_tools(*names: str) -> trajectory_lib.Trajectory:
      return _trajectory([
          _step(
              1,
              trajectory_lib.Source.AGENT,
              tool_calls=[
                  trajectory_lib.ToolCall(
                      tool_call_id=str(i), function_name=name
                  )
                  for i, name in enumerate(names)
              ],
          )
      ])

    run = stats_lib.RunStats.from_trajectories(
        [with_tools("b", "c"), with_tools("c", "a")]
    )

    self.assertEqual(run.tool_counts, {"c": 2, "a": 1, "b": 1})
    self.assertEqual(list(run.tool_counts), ["b", "c", "a"])
    self.assertEqual(list(run.to_json_dict()["tool_counts"]), ["c", "b", "a"])

  def test_run_json(self) -> None:
    run = stats_lib.RunStats.from_trajectories([
        _tunix_trajectory(
            steps=[
                trajectory_lib.TunixAgentStep(
                    step_id=0, source=trajectory_lib.Source.AGENT, message="m"
                )
            ],
            total_reward=1.0,
            status="SUCCEEDED",
            final_metrics=trajectory_lib.FinalMetrics(
                total_prompt_tokens=5, total_completion_tokens=2
            ),
        ),
        _tunix_trajectory(),
    ])

    self.assertEqual(
        run.to_json_dict(),
        {
            "num_trajectories": 2,
            "steps": {
                "total": 1,
                "agent": 1,
                "user": 0,
                "system": 0,
                "other": 0,
            },
            "tokens": {"prompt": 5, "completion": 2, "cached": 0, "total": 7},
            "num_with_tokens": 1,
            "status_counts": {"SUCCEEDED": 1},
            "num_with_status": 1,
            "mean_reward": 1.0,
            "num_with_reward": 1,
            "tool_counts": {},
        },
    )


class JsonDictTest(absltest.TestCase):

  def test_unreported_quantities_serialize_as_null(self) -> None:
    trajectory = trajectory_lib.TunixTrajectory(agent=_AGENT, steps=[])

    payload = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).to_json_dict()

    self.assertEqual(
        payload,
        {
            "trajectory_id": None,
            "status": None,
            "total_reward": None,
            "steps": {
                "total": 0,
                "agent": 0,
                "user": 0,
                "system": 0,
                "other": 0,
            },
            "tokens": None,
            "tool_call_names": [],
        },
    )

  def test_stats_serialize_every_reported_quantity(self) -> None:
    trajectory = _tunix_trajectory(
        status="SUCCEEDED",
        total_reward=0.85,
        steps=[
            trajectory_lib.TunixAgentStep(
                step_id=0,
                source=trajectory_lib.Source.AGENT,
                message="m",
                tool_calls=[
                    trajectory_lib.ToolCall(
                        tool_call_id="0", function_name="find_files"
                    )
                ],
            ),
            trajectory_lib.TunixEnvStep(
                step_id=1, source=trajectory_lib.Source.USER, message="m"
            ),
        ],
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=520,
            total_completion_tokens=90,
            total_cached_tokens=220,
        ),
    )

    payload = stats_lib.TrajectoryStats.from_trajectory(
        trajectory
    ).to_json_dict()

    self.assertEqual(
        payload,
        {
            "trajectory_id": "traj_test",
            "status": "SUCCEEDED",
            "total_reward": 0.85,
            "steps": {
                "total": 2,
                "agent": 1,
                "user": 1,
                "system": 0,
                "other": 0,
            },
            "tokens": {
                "prompt": 520,
                "completion": 90,
                "cached": 220,
                "total": 610,
            },
            "tool_call_names": ["find_files"],
        },
    )


if __name__ == "__main__":
  absltest.main()
