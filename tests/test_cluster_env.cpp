#include "cluster_env.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}  // namespace

int main() {
    const std::vector<hpc::NodeCapacity> nodes{{4.0F, 16.0F}, {8.0F, 32.0F}};
    hpc::RewardConfig rewards;
    rewards.time_scale = 10.0F;
    hpc::ClusterEnv environment(nodes, 3, rewards);
    const std::vector<hpc::Job> jobs{
        {0, 0.0, 2.0F, 4.0F, 3.0F, 10.0F},
        {1, 0.0, 6.0F, 8.0F, 2.0F, 10.0F},
        {2, 5.0, 1.0F, 2.0F, 1.0F, 10.0F},
    };
    const auto& initial = environment.reset(jobs, 42);
    require(initial.size() == 2 * hpc::ClusterEnv::kNodeFeatureCount +
                                  3 * hpc::ClusterEnv::kJobFeatureCount +
                                  hpc::ClusterEnv::kGlobalFeatureCount,
            "unexpected state width");
    require(environment.action_count() == 7, "unexpected action count");
    require(environment.is_action_valid(0, 0), "job 0 should fit node 0");
    require(!environment.is_action_valid(1, 0), "job 1 must not fit node 0");
    require(environment.is_action_valid(1, 1), "job 1 should fit node 1");
    require(environment.action_mask().back() == 1, "wait action should be enabled");

    const hpc::StepResult first = environment.step(0, 0);
    require(first.action_valid, "first action rejected");
    require(!first.terminated, "episode ended too soon");
    const hpc::StepResult second = environment.step(1, 1);
    require(second.action_valid, "second action rejected");
    require(!second.terminated, "episode ended before future arrival");
    require(std::abs(environment.current_time() - 2.0) < 1.0e-6,
            "engine did not advance to first completion");

    std::size_t step_count = 0;
    while (!environment.is_terminal()) {
        require(++step_count < 20, "episode failed to make progress");
        const auto mask = environment.action_mask();
        bool scheduled = false;
        for (std::size_t action = 0; action + 1 < mask.size(); ++action) {
            if (mask[action] != 0) {
                const int job = static_cast<int>(action / environment.node_count());
                const int node = static_cast<int>(action % environment.node_count());
                environment.step(job, node);
                scheduled = true;
                break;
            }
        }
        if (!scheduled) {
            environment.step(-1, -1);
        }
    }
    require(environment.completed_jobs() == jobs.size(), "not all jobs completed");
    require(environment.sla_violations() == 0, "unexpected SLA violation");
    require(!environment.render().empty(), "render output was empty");

    bool rejected = false;
    try {
        environment.reset({{0, 0.0, 100.0F, 1.0F, 1.0F, 1.0F}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "oversized job was accepted");
    std::cout << "cluster_env_test passed\n";
    return 0;
}
