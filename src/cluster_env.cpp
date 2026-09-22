#include "cluster_env.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace hpc {
namespace {
constexpr float kEpsilon = 1.0e-5F;

float safe_ratio(const float numerator, const float denominator) noexcept {
    return denominator > kEpsilon ? numerator / denominator : 0.0F;
}
}  // namespace

ClusterEnv::ClusterEnv(std::vector<NodeCapacity> capacities,
                       const std::size_t queue_capacity,
                       RewardConfig reward_config)
    : queue_capacity_(queue_capacity), reward_config_(reward_config) {
    nodes_.reserve(capacities.size());
    for (const auto capacity : capacities) {
        nodes_.push_back(NodeState{capacity, 0.0F, 0.0F, {}});
        max_cpu_ = std::max(max_cpu_, capacity.cpu);
        max_memory_ = std::max(max_memory_, capacity.memory);
    }
    validate_configuration();
    if (nodes_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        queue_capacity_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("node and queue counts must fit in the action index type");
    }
    if (queue_capacity_ >
        (std::numeric_limits<std::size_t>::max() - 1) / nodes_.size()) {
        throw std::length_error("queue_capacity makes the action space too large");
    }
    if (queue_capacity_ >
        (std::numeric_limits<std::size_t>::max() -
         nodes_.size() * kNodeFeatureCount - kGlobalFeatureCount) /
            kJobFeatureCount) {
        throw std::length_error("queue_capacity makes the state vector too large");
    }
    ready_queue_.resize(queue_capacity_);
    state_.resize(nodes_.size() * kNodeFeatureCount +
                  queue_capacity_ * kJobFeatureCount + kGlobalFeatureCount,
                  0.0F);
    update_state();
}

void ClusterEnv::validate_configuration() const {
    if (nodes_.empty()) {
        throw std::invalid_argument("ClusterEnv requires at least one node");
    }
    if (queue_capacity_ == 0) {
        throw std::invalid_argument("queue_capacity must be greater than zero");
    }
    for (const auto& node : nodes_) {
        if (!std::isfinite(node.capacity.cpu) || !std::isfinite(node.capacity.memory) ||
            !(node.capacity.cpu > 0.0F) || !(node.capacity.memory > 0.0F)) {
            throw std::invalid_argument("node CPU and memory capacities must be positive");
        }
    }
    if (!std::isfinite(reward_config_.alpha) || !std::isfinite(reward_config_.beta) ||
        !std::isfinite(reward_config_.gamma) ||
        !std::isfinite(reward_config_.invalid_action_penalty) ||
        !std::isfinite(reward_config_.time_scale) || reward_config_.alpha < 0.0F ||
        reward_config_.beta < 0.0F || reward_config_.gamma < 0.0F ||
        reward_config_.invalid_action_penalty < 0.0F ||
        !(reward_config_.time_scale > 0.0F)) {
        throw std::invalid_argument(
            "reward coefficients must be finite/nonnegative and time_scale positive");
    }
}

const std::vector<float>& ClusterEnv::reset(std::vector<Job> workload, const std::uint64_t seed) {
    (void)seed;  // The engine is deterministic; retained for Gymnasium compatibility.
    for (const auto& job : workload) {
        if (!std::isfinite(job.submission_time) || !std::isfinite(job.cpu_request) ||
            !std::isfinite(job.memory_request) || !std::isfinite(job.duration) ||
            !std::isfinite(job.sla_wait) || job.submission_time < 0.0 ||
            !(job.cpu_request > 0.0F) ||
            !(job.memory_request > 0.0F) || !(job.duration > 0.0F) ||
            job.sla_wait < 0.0F) {
            throw std::invalid_argument("workload contains an invalid job field");
        }
        const bool fits_a_node = std::any_of(nodes_.begin(), nodes_.end(),
                                             [&job](const NodeState& node) {
            return job.cpu_request <= node.capacity.cpu + kEpsilon &&
                   job.memory_request <= node.capacity.memory + kEpsilon;
        });
        if (!fits_a_node) {
            throw std::invalid_argument("a job cannot fit on any configured node");
        }
    }
    std::stable_sort(workload.begin(), workload.end(), [](const Job& lhs, const Job& rhs) {
        return lhs.submission_time < rhs.submission_time;
    });
    workload_ = std::move(workload);
    next_arrival_ = 0;
    completed_jobs_ = 0;
    sla_violations_ = 0;
    current_time_ = workload_.empty() ? 0.0 : workload_.front().submission_time;
    terminal_ = workload_.empty();
    std::fill(ready_queue_.begin(), ready_queue_.end(), std::nullopt);
    for (auto& node : nodes_) {
        node.used_cpu = 0.0F;
        node.used_memory = 0.0F;
        node.running.clear();
    }
    admit_ready_jobs();
    update_state();
    return state_;
}

void ClusterEnv::release_finished_jobs() {
    for (auto& node : nodes_) {
        auto out = node.running.begin();
        for (auto it = node.running.begin(); it != node.running.end(); ++it) {
            if (it->finish_time <= current_time_ + static_cast<double>(kEpsilon)) {
                node.used_cpu = std::max(0.0F, node.used_cpu - it->cpu);
                node.used_memory = std::max(0.0F, node.used_memory - it->memory);
                ++completed_jobs_;
            } else {
                *out++ = *it;
            }
        }
        node.running.erase(out, node.running.end());
    }
}

void ClusterEnv::admit_ready_jobs() {
    while (next_arrival_ < workload_.size() &&
           workload_[next_arrival_].submission_time <= current_time_ + kEpsilon) {
        const auto empty_slot = std::find_if(ready_queue_.begin(), ready_queue_.end(),
                                             [](const auto& item) { return !item.has_value(); });
        if (empty_slot == ready_queue_.end()) {
            break;
        }
        *empty_slot = workload_[next_arrival_++];
    }
}

bool ClusterEnv::is_action_valid(const int job_idx, const int node_idx) const noexcept {
    if (terminal_ || job_idx < 0 || node_idx < 0 ||
        static_cast<std::size_t>(job_idx) >= queue_capacity_ ||
        static_cast<std::size_t>(node_idx) >= nodes_.size()) {
        return false;
    }
    const auto& queued = ready_queue_[static_cast<std::size_t>(job_idx)];
    if (!queued) {
        return false;
    }
    const auto& node = nodes_[static_cast<std::size_t>(node_idx)];
    return node.used_cpu + queued->cpu_request <= node.capacity.cpu + kEpsilon &&
           node.used_memory + queued->memory_request <= node.capacity.memory + kEpsilon;
}

bool ClusterEnv::has_feasible_action() const noexcept {
    for (std::size_t job = 0; job < queue_capacity_; ++job) {
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            if (is_action_valid(static_cast<int>(job), static_cast<int>(node))) {
                return true;
            }
        }
    }
    return false;
}

std::optional<double> ClusterEnv::next_event_time() const noexcept {
    double next = std::numeric_limits<double>::infinity();
    if (next_arrival_ < workload_.size() &&
        std::any_of(ready_queue_.begin(), ready_queue_.end(),
                    [](const auto& item) { return !item.has_value(); })) {
        next = std::max(current_time_, workload_[next_arrival_].submission_time);
    }
    for (const auto& node : nodes_) {
        for (const auto& running : node.running) {
            if (running.finish_time > current_time_ + kEpsilon) {
                next = std::min(next, running.finish_time);
            }
        }
    }
    if (std::isfinite(next)) {
        return next;
    }
    return std::nullopt;
}

void ClusterEnv::advance_to_next_event(const bool force_advance) {
    release_finished_jobs();
    admit_ready_jobs();
    if (!force_advance && has_feasible_action()) {
        return;
    }
    if (const auto next = next_event_time(); next.has_value()) {
        current_time_ = std::max(current_time_, *next);
        release_finished_jobs();
        admit_ready_jobs();
    }
}

float ClusterEnv::mean_utilization() const noexcept {
    float sum = 0.0F;
    for (const auto& node : nodes_) {
        sum += 0.5F * (safe_ratio(node.used_cpu, node.capacity.cpu) +
                       safe_ratio(node.used_memory, node.capacity.memory));
    }
    return sum / static_cast<float>(nodes_.size());
}

float ClusterEnv::normalized_queue_wait() const noexcept {
    float sum = 0.0F;
    std::size_t count = 0;
    for (const auto& queued : ready_queue_) {
        if (queued) {
            sum += static_cast<float>(std::max(0.0, current_time_ - queued->submission_time)) /
                   reward_config_.time_scale;
            ++count;
        }
    }
    return count == 0 ? 0.0F : sum / static_cast<float>(count);
}

std::size_t ClusterEnv::current_sla_violations() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        ready_queue_.begin(), ready_queue_.end(), [this](const auto& queued) {
            return queued && queued->sla_wait > 0.0F &&
                   current_time_ - queued->submission_time > queued->sla_wait;
        }));
}

StepResult ClusterEnv::step(const int job_idx, const int node_idx) {
    if (terminal_) {
        throw std::logic_error("step() called on a terminal environment; call reset()");
    }
    const bool wait_action = job_idx == -1 && node_idx == -1;
    const bool valid = is_action_valid(job_idx, node_idx);
    float invalid_penalty = 0.0F;

    if (valid) {
        auto& queued = ready_queue_[static_cast<std::size_t>(job_idx)];
        auto& node = nodes_[static_cast<std::size_t>(node_idx)];
        const Job job = *queued;
        node.used_cpu += job.cpu_request;
        node.used_memory += job.memory_request;
        node.running.push_back(
            RunningJob{job.cpu_request, job.memory_request, current_time_ + job.duration});
        if (job.sla_wait > 0.0F && current_time_ - job.submission_time > job.sla_wait) {
            ++sla_violations_;
        }
        queued.reset();
        admit_ready_jobs();
    } else if (!wait_action) {
        invalid_penalty = reward_config_.invalid_action_penalty;
    }

    advance_to_next_event(wait_action || !valid);
    const std::size_t active_sla_violations = current_sla_violations();

    // R = -alpha * mean_normalized_wait + beta * mean_resource_utilization
    //     -gamma * active_SLA_violations - invalid_action_penalty.
    const float reward = -reward_config_.alpha * normalized_queue_wait() +
                         reward_config_.beta * mean_utilization() -
                         reward_config_.gamma * static_cast<float>(active_sla_violations) -
                         invalid_penalty;

    const bool queue_empty = std::none_of(ready_queue_.begin(), ready_queue_.end(),
                                          [](const auto& item) { return item.has_value(); });
    const bool no_running = std::all_of(nodes_.begin(), nodes_.end(),
                                        [](const auto& node) { return node.running.empty(); });
    terminal_ = next_arrival_ == workload_.size() && queue_empty && no_running;
    update_state();
    return StepResult{reward, terminal_, false, valid || wait_action, completed_jobs_,
                      sla_violations_, current_time_};
}

std::vector<std::uint8_t> ClusterEnv::action_mask() const {
    std::vector<std::uint8_t> mask(action_count(), 0);
    for (std::size_t job = 0; job < queue_capacity_; ++job) {
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            mask[job * nodes_.size() + node] = static_cast<std::uint8_t>(
                is_action_valid(static_cast<int>(job), static_cast<int>(node)));
        }
    }
    // WAIT is meaningful only when an arrival or completion can advance time.
    // Masking it in an idle dead-end prevents learned policies from stalling forever.
    mask.back() = static_cast<std::uint8_t>(!terminal_ && next_event_time().has_value());
    return mask;
}

void ClusterEnv::update_state() {
    std::fill(state_.begin(), state_.end(), 0.0F);
    std::size_t offset = 0;
    for (const auto& node : nodes_) {
        state_[offset++] = safe_ratio(node.used_cpu, node.capacity.cpu);
        state_[offset++] = safe_ratio(node.used_memory, node.capacity.memory);
        state_[offset++] = safe_ratio(node.capacity.cpu, max_cpu_);
        state_[offset++] = safe_ratio(node.capacity.memory, max_memory_);
        state_[offset++] = std::min(1.0F, static_cast<float>(node.running.size()) / 32.0F);
    }
    for (const auto& queued : ready_queue_) {
        if (queued) {
            state_[offset++] = safe_ratio(queued->cpu_request, max_cpu_);
            state_[offset++] = safe_ratio(queued->memory_request, max_memory_);
            state_[offset++] = queued->duration / reward_config_.time_scale;
            const float wait = static_cast<float>(
                std::max(0.0, current_time_ - queued->submission_time));
            state_[offset++] = wait / reward_config_.time_scale;
            state_[offset++] = (queued->sla_wait - wait) / reward_config_.time_scale;
            state_[offset++] = 1.0F;
        } else {
            offset += kJobFeatureCount;
        }
    }
    state_[offset++] = static_cast<float>(current_time_ / reward_config_.time_scale);
    state_[offset++] = workload_.empty()
                           ? 1.0F
                           : static_cast<float>(next_arrival_) / static_cast<float>(workload_.size());
    state_[offset++] = workload_.empty()
                           ? 1.0F
                           : static_cast<float>(completed_jobs_) /
                                 static_cast<float>(workload_.size());
    state_[offset] = mean_utilization();
}

const std::vector<float>& ClusterEnv::get_state_vector() const noexcept { return state_; }

bool ClusterEnv::is_terminal() const noexcept { return terminal_; }

std::string ClusterEnv::render() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "t=" << current_time_ << " nodes=[";
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        const auto& node = nodes_[index];
        if (index > 0) {
            output << ", ";
        }
        output << index << ":cpu " << node.used_cpu << '/' << node.capacity.cpu << ",mem "
               << node.used_memory << '/' << node.capacity.memory;
    }
    const auto queued = std::count_if(ready_queue_.begin(), ready_queue_.end(),
                                      [](const auto& item) { return item.has_value(); });
    output << "] queued=" << queued << " completed=" << completed_jobs_;
    return output.str();
}

}  // namespace hpc
