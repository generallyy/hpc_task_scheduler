#include "cluster_env.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
std::vector<hpc::NodeCapacity> parse_nodes(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& values) {
    const auto buffer = values.request();
    if (buffer.ndim != 2 || buffer.shape[1] != 2 || buffer.shape[0] < 1) {
        throw std::invalid_argument("node_capacities must have shape [N, 2]");
    }
    const auto data = values.unchecked<2>();
    std::vector<hpc::NodeCapacity> nodes;
    nodes.reserve(static_cast<std::size_t>(buffer.shape[0]));
    for (py::ssize_t row = 0; row < buffer.shape[0]; ++row) {
        nodes.push_back({data(row, 0), data(row, 1)});
    }
    return nodes;
}

std::vector<hpc::Job> parse_jobs(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values) {
    const auto buffer = values.request();
    if (buffer.ndim != 2 || buffer.shape[1] != 5) {
        throw std::invalid_argument(
            "workload must have shape [jobs, 5]: submission,cpu,memory,duration,sla_wait");
    }
    const auto data = values.unchecked<2>();
    std::vector<hpc::Job> jobs;
    jobs.reserve(static_cast<std::size_t>(buffer.shape[0]));
    for (py::ssize_t row = 0; row < buffer.shape[0]; ++row) {
        jobs.push_back(hpc::Job{static_cast<std::int64_t>(row), data(row, 0),
                                static_cast<float>(data(row, 1)),
                                static_cast<float>(data(row, 2)),
                                static_cast<float>(data(row, 3)),
                                static_cast<float>(data(row, 4))});
    }
    return jobs;
}

py::array_t<float> copy_state(const hpc::ClusterEnv& environment) {
    const auto& state = environment.get_state_vector();
    py::array_t<float> result(static_cast<py::ssize_t>(state.size()));
    std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(float));
    return result;
}

py::dict step_info(const hpc::StepResult& result) {
    py::dict info;
    info["action_valid"] = result.action_valid;
    info["completed_jobs"] = result.completed_jobs;
    info["sla_violations"] = result.sla_violations;
    info["current_time"] = result.current_time;
    return info;
}
}  // namespace

PYBIND11_MODULE(_cluster_env, module) {
    module.doc() = "Low-latency event-driven HPC scheduling environment";
    module.attr("NODE_FEATURE_COUNT") = hpc::ClusterEnv::kNodeFeatureCount;
    module.attr("JOB_FEATURE_COUNT") = hpc::ClusterEnv::kJobFeatureCount;
    module.attr("GLOBAL_FEATURE_COUNT") = hpc::ClusterEnv::kGlobalFeatureCount;

    py::class_<hpc::RewardConfig>(module, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("alpha", &hpc::RewardConfig::alpha)
        .def_readwrite("beta", &hpc::RewardConfig::beta)
        .def_readwrite("gamma", &hpc::RewardConfig::gamma)
        .def_readwrite("invalid_action_penalty", &hpc::RewardConfig::invalid_action_penalty)
        .def_readwrite("time_scale", &hpc::RewardConfig::time_scale);

    py::class_<hpc::ClusterEnv>(module, "ClusterEnv")
        .def(py::init([](const py::array_t<float, py::array::c_style |
                                                   py::array::forcecast>& capacities,
                         const std::size_t queue_capacity,
                         const hpc::RewardConfig& reward_config) {
                 return hpc::ClusterEnv(parse_nodes(capacities), queue_capacity, reward_config);
             }),
             py::arg("node_capacities"), py::arg("queue_capacity"),
             py::arg("reward_config") = hpc::RewardConfig{})
        .def("reset",
             [](hpc::ClusterEnv& environment,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& workload,
                const std::uint64_t seed) {
                 environment.reset(parse_jobs(workload), seed);
                 return py::make_tuple(copy_state(environment), py::dict());
             },
             py::arg("workload"), py::arg("seed") = 0)
        .def("step",
             [](hpc::ClusterEnv& environment, const int job_idx, const int node_idx) {
                 const hpc::StepResult result = environment.step(job_idx, node_idx);
                 return py::make_tuple(copy_state(environment), result.reward, result.terminated,
                                       result.truncated, step_info(result));
             },
             py::arg("job_idx"), py::arg("node_idx"))
        .def("render", &hpc::ClusterEnv::render)
        .def("is_terminal", &hpc::ClusterEnv::is_terminal)
        .def("is_action_valid", &hpc::ClusterEnv::is_action_valid)
        .def("action_mask", [](const hpc::ClusterEnv& environment) {
            const auto mask = environment.action_mask();
            py::array_t<bool> result(static_cast<py::ssize_t>(mask.size()));
            auto output = result.mutable_unchecked<1>();
            for (py::ssize_t index = 0; index < output.shape(0); ++index) {
                output(index) = mask[static_cast<std::size_t>(index)] != 0;
            }
            return result;
        })
        .def("get_state_vector", &copy_state,
             "Return an owned observation copy safe across subsequent steps.")
        .def("state_view", [](hpc::ClusterEnv& environment) {
            const auto& state = environment.get_state_vector();
            return py::array_t<float>(
                {static_cast<py::ssize_t>(state.size())},
                {static_cast<py::ssize_t>(sizeof(float))}, state.data(),
                py::cast(&environment, py::return_value_policy::reference));
        }, "Return a read-only-lifetime zero-copy view invalidated by the next state update.")
        .def_property_readonly("state_size", &hpc::ClusterEnv::state_size)
        .def_property_readonly("action_count", &hpc::ClusterEnv::action_count)
        .def_property_readonly("node_count", &hpc::ClusterEnv::node_count)
        .def_property_readonly("queue_capacity", &hpc::ClusterEnv::queue_capacity)
        .def_property_readonly("current_time", &hpc::ClusterEnv::current_time)
        .def_property_readonly("completed_jobs", &hpc::ClusterEnv::completed_jobs)
        .def_property_readonly("sla_violations", &hpc::ClusterEnv::sla_violations);
}
