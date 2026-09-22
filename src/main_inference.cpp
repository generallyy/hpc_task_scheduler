#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Arguments {
    std::filesystem::path model;
    std::size_t iterations{10'000};
    std::size_t warmup{200};
};

Arguments parse_arguments(const int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: hpc_inference <model.onnx> [--iterations N] [--warmup N]");
    }
    Arguments arguments{std::filesystem::path(argv[1])};
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        if (option == "--iterations") {
            arguments.iterations = std::stoull(argv[++index]);
        } else if (option == "--warmup") {
            arguments.warmup = std::stoull(argv[++index]);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.iterations == 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    return arguments;
}

std::size_t fixed_dimension(const std::vector<std::int64_t>& shape,
                            const std::size_t axis,
                            const std::string& tensor_name) {
    if (axis >= shape.size() || shape[axis] <= 0) {
        throw std::runtime_error(tensor_name + " must have a fixed non-batch dimension");
    }
    return static_cast<std::size_t>(shape[axis]);
}

double percentile(std::vector<double> values, const double quantile) {
    const std::size_t index = static_cast<std::size_t>(
        std::floor(quantile * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        if (!std::filesystem::is_regular_file(arguments.model)) {
            throw std::runtime_error("model does not exist: " + arguments.model.string());
        }

        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "hpc-scheduler");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        Ort::Session session(environment, arguments.model.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        if (session.GetInputCount() != 2 || session.GetOutputCount() != 1) {
            throw std::runtime_error("expected state/action_mask inputs and one logits output");
        }
        const auto mask_name = session.GetInputNameAllocated(1, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const auto state_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto mask_info = session.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo();
        const auto output_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        if (state_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            mask_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("expected float32 state/logits and uint8 action mask");
        }
        const std::vector<std::int64_t> input_shape = state_info.GetShape();
        const std::vector<std::int64_t> output_shape = output_info.GetShape();
        const std::vector<std::int64_t> mask_shape = mask_info.GetShape();
        if (input_shape.size() != 2 || mask_shape.size() != 2 || output_shape.size() != 2) {
            throw std::runtime_error("expected rank-2 state, action-mask, and action-logit tensors");
        }
        const std::size_t state_size = fixed_dimension(input_shape, 1, "state input");
        const std::size_t action_count = fixed_dimension(output_shape, 1, "action output");
        if (fixed_dimension(mask_shape, 1, "action mask") != action_count) {
            throw std::runtime_error("action mask and model output widths differ");
        }
        std::vector<float> state(state_size);
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
        std::generate(state.begin(), state.end(), [&] { return distribution(generator); });
        const std::array<std::int64_t, 2> concrete_shape{
            1, static_cast<std::int64_t>(state_size)};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, state.data(), state.size(), concrete_shape.data(), concrete_shape.size());
        std::vector<std::uint8_t> action_mask(action_count, 1);
        const std::array<std::int64_t, 2> mask_concrete_shape{
            1, static_cast<std::int64_t>(action_count)};
        Ort::Value mask_input = Ort::Value::CreateTensor<std::uint8_t>(
            memory, action_mask.data(), action_mask.size(), mask_concrete_shape.data(),
            mask_concrete_shape.size());
        const char* input_names[] = {input_name.get(), mask_name.get()};
        const char* output_names[] = {output_name.get()};
        std::array<Ort::Value, 2> inputs{std::move(input), std::move(mask_input)};
        std::vector<float> logits(action_count, 0.0F);
        const std::array<std::int64_t, 2> output_concrete_shape{
            1, static_cast<std::int64_t>(action_count)};
        Ort::Value output = Ort::Value::CreateTensor<float>(
            memory, logits.data(), logits.size(), output_concrete_shape.data(),
            output_concrete_shape.size());
        Ort::RunOptions run_options;

        auto infer = [&]() {
            session.Run(run_options, input_names, inputs.data(), inputs.size(), output_names,
                        &output, 1);
        };
        for (std::size_t index = 0; index < arguments.warmup; ++index) {
            infer();
        }

        std::vector<double> latencies_us;
        latencies_us.reserve(arguments.iterations);
        std::size_t selected_action = 0;
        for (std::size_t index = 0; index < arguments.iterations; ++index) {
            const auto start = std::chrono::steady_clock::now();
            infer();
            selected_action = static_cast<std::size_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            const auto end = std::chrono::steady_clock::now();
            latencies_us.push_back(
                std::chrono::duration<double, std::micro>(end - start).count());
            state[index % state.size()] = std::nextafter(state[index % state.size()], 1.0F);
        }
        const double mean = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
                            static_cast<double>(latencies_us.size());
        const double p50 = percentile(latencies_us, 0.50);
        const double p95 = percentile(latencies_us, 0.95);
        const double p99 = percentile(latencies_us, 0.99);
        std::cout << "state_size=" << state_size << " action_count=" << action_count
                  << " selected_action=" << selected_action << '\n'
                  << "latency_us mean=" << mean << " p50=" << p50 << " p95=" << p95
                  << " p99=" << p99 << '\n'
                  << "target_p95_lt_200us=" << (p95 < 200.0 ? "true" : "false") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "inference error: " << error.what() << '\n';
        return 1;
    }
}
