#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hpc {

struct NodeCapacity {
    float cpu{0.0F};
    float memory{0.0F};
};

struct Job {
    std::int64_t id{0};
    double submission_time{0.0};
    float cpu_request{0.0F};
    float memory_request{0.0F};
    float duration{0.0F};
    float sla_wait{0.0F};
};

struct RewardConfig {
    float alpha{1.0F};
    float beta{0.25F};
    float gamma{2.0F};
    float invalid_action_penalty{2.0F};
    float time_scale{3600.0F};
};

struct StepResult {
    float reward{0.0F};
    bool terminated{false};
    bool truncated{false};
    bool action_valid{false};
    std::size_t completed_jobs{0};
    std::size_t sla_violations{0};
    double current_time{0.0};
};

/**
 * Event-driven heterogeneous-cluster scheduling environment.
 *
 * The observation is kept in one aligned, contiguous float vector:
 * [node features][queue-slot features][global features]. This makes reads
 * cache-friendly and permits pybind11 to publish a zero-copy NumPy view.
 */
class ClusterEnv final {
public:
    static constexpr std::size_t kNodeFeatureCount = 5;
    static constexpr std::size_t kJobFeatureCount = 6;
    static constexpr std::size_t kGlobalFeatureCount = 4;

    ClusterEnv(std::vector<NodeCapacity> capacities,
               std::size_t queue_capacity,
               RewardConfig reward_config = {});

    const std::vector<float>& reset(std::vector<Job> workload, std::uint64_t seed = 0);
    StepResult step(int job_idx, int node_idx);

    [[nodiscard]] const std::vector<float>& get_state_vector() const noexcept;
    [[nodiscard]] std::vector<std::uint8_t> action_mask() const;
    [[nodiscard]] bool is_action_valid(int job_idx, int node_idx) const noexcept;
    [[nodiscard]] bool is_terminal() const noexcept;
    [[nodiscard]] std::string render() const;

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t queue_capacity() const noexcept { return queue_capacity_; }
    [[nodiscard]] std::size_t state_size() const noexcept { return state_.size(); }
    [[nodiscard]] std::size_t action_count() const noexcept {
        return nodes_.size() * queue_capacity_ + 1;
    }
    [[nodiscard]] double current_time() const noexcept { return current_time_; }
    [[nodiscard]] std::size_t completed_jobs() const noexcept { return completed_jobs_; }
    [[nodiscard]] std::size_t sla_violations() const noexcept { return sla_violations_; }

private:
    struct RunningJob {
        float cpu{0.0F};
        float memory{0.0F};
        double finish_time{0.0};
    };

    struct NodeState {
        NodeCapacity capacity{};
        float used_cpu{0.0F};
        float used_memory{0.0F};
        std::vector<RunningJob> running{};
    };

    std::vector<NodeState> nodes_;
    std::size_t queue_capacity_;
    RewardConfig reward_config_;
    std::vector<Job> workload_;
    std::size_t next_arrival_{0};
    std::vector<std::optional<Job>> ready_queue_;
    std::vector<float> state_;
    double current_time_{0.0};
    std::size_t completed_jobs_{0};
    std::size_t sla_violations_{0};
    bool terminal_{true};
    float max_cpu_{1.0F};
    float max_memory_{1.0F};

    void validate_configuration() const;
    void release_finished_jobs();
    void admit_ready_jobs();
    void update_state();
    void advance_to_next_event(bool force_advance);
    [[nodiscard]] bool has_feasible_action() const noexcept;
    [[nodiscard]] std::optional<double> next_event_time() const noexcept;
    [[nodiscard]] float mean_utilization() const noexcept;
    [[nodiscard]] float normalized_queue_wait() const noexcept;
    [[nodiscard]] std::size_t current_sla_violations() const noexcept;
};

}  // namespace hpc
