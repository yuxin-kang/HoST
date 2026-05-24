#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl_getup_example/host_policy_contract.h"

namespace {

void RequireNear(double actual, double expected, const char* label) {
  if (std::fabs(actual - expected) > 1e-12) {
    throw std::runtime_error(std::string(label) + ": " + std::to_string(actual) +
                             " != " + std::to_string(expected));
  }
}

void RequireVectorNear(const std::vector<double>& actual,
                       const std::vector<double>& expected,
                       const char* label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(label) + " size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    RequireNear(actual[i], expected[i], label);
  }
}

}  // namespace

int main() {
  using runner::rl_getup_contract::BuildStepObservation;
  using runner::rl_getup_contract::ComputeRelativeJointTargets;
  using runner::rl_getup_contract::ObservationScales;
  using runner::rl_getup_contract::UpdateObservationHistory;

  constexpr int kNumActions = 23;
  constexpr int kStepDim = 76;
  constexpr int kHistorySteps = 6;
  constexpr double kActionRescale = 0.6;
  const ObservationScales scales{.angular_velocity = 0.25, .dof_position = 1.0, .dof_velocity = 0.05};

  std::vector<double> q(kNumActions);
  std::vector<double> qd(kNumActions);
  std::vector<double> previous_action(kNumActions);
  std::vector<double> raw_action(kNumActions);
  std::vector<double> action_scale(kNumActions);
  for (int i = 0; i < kNumActions; ++i) {
    q[static_cast<std::size_t>(i)] = -0.5 + 0.05 * i;
    qd[static_cast<std::size_t>(i)] = 0.3 - 0.02 * i;
    previous_action[static_cast<std::size_t>(i)] = -1.0 + 2.0 * i / (kNumActions - 1);
    raw_action[static_cast<std::size_t>(i)] = -0.2 + 0.01 * i;
    action_scale[static_cast<std::size_t>(i)] = 0.6 + 0.02 * i;
  }

  const std::vector<double> first_step = BuildStepObservation(
      {0.2, -0.4, 0.6}, {0.01, -0.02, -0.999}, q, qd, previous_action, kActionRescale, scales);
  if (first_step.size() != kStepDim) {
    throw std::runtime_error("first step observation size mismatch");
  }
  RequireNear(first_step[0], 0.2 * scales.angular_velocity, "base angular velocity x");
  RequireNear(first_step[3], 0.01, "projected gravity x");
  RequireNear(first_step[6], q[0] * scales.dof_position, "joint position start");
  RequireNear(first_step[29], qd[0] * scales.dof_velocity, "joint velocity start");
  RequireNear(first_step[52], previous_action[0], "previous action start");
  RequireNear(first_step[75], kActionRescale, "action rescale");

  std::vector<double> history(static_cast<std::size_t>(kStepDim * kHistorySteps), 0.0);
  UpdateObservationHistory(history, kStepDim, kHistorySteps, first_step, 100.0, false);
  std::vector<double> expected_after_first(history.size(), 0.0);
  std::copy(first_step.begin(), first_step.end(), expected_after_first.begin() + 5 * kStepDim);
  RequireVectorNear(history, expected_after_first, "first history update");

  std::vector<double> second_step = first_step;
  second_step[0] = 200.0;
  second_step[75] = 0.4;
  UpdateObservationHistory(history, kStepDim, kHistorySteps, second_step, 10.0, true);
  std::vector<double> expected_after_second(history.size(), 0.0);
  std::copy(first_step.begin(), first_step.end(), expected_after_second.begin() + 4 * kStepDim);
  std::copy(second_step.begin(), second_step.end(), expected_after_second.begin() + 5 * kStepDim);
  expected_after_second[5 * kStepDim] = 10.0;
  RequireVectorNear(history, expected_after_second, "second history update");

  const std::vector<double> targets = ComputeRelativeJointTargets(q, raw_action, action_scale, kActionRescale);
  if (targets.size() != kNumActions) {
    throw std::runtime_error("target size mismatch");
  }
  for (int i = 0; i < kNumActions; ++i) {
    const std::size_t index = static_cast<std::size_t>(i);
    RequireNear(targets[index], q[index] + raw_action[index] * action_scale[index] * kActionRescale,
                "relative q_des target");
  }

  std::cout << "engineai rl_getup contract fixture passed\n";
  return 0;
}
