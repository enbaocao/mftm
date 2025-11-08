#pragma once

namespace megakernel {
namespace generic {

// Tag types to select worker role at dispatch time
struct worker_consumer {};
struct worker_loader {};
struct worker_storer {};
struct worker_launcher {};

} // namespace generic
} // namespace megakernel

