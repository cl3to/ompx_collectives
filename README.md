# OMPX Collectives

This repository contains an experimental implementation of **OMPX Collectives**, an extension of OpenMP aimed at supporting high-level collective operations (e.g., broadcast, gather, reduce) in heterogeneous systems using target offloading.

The implementation focuses on enabling efficient collective operations across device and host threads, providing abstractions for data movement and synchronization in offloaded regions.

The implementation includes:
- `ompx_*` applications: Use a custom collective runtime (`ompx_collectives.c`)
- `omp_*` applications: Standalone OpenMP implementations for baseline comparison

## Features

- 🔄 Collective operations for OpenMP target model
- 🚀 Optimized data movement for GPUs using topology-aware algorithms


## Motivation

Traditional OpenMP lacks high-level support for collective operations across devices. This project explores how such functionality can be added as a natural extension, facilitating the implementation of distributed parallel algorithms on offloaded regions.


## 🔧 Build Instructions

### Requirements

- GCC or Clang with OpenMP support
- Make
- Optional: GPU offloading support (`-fopenmp-targets` with Clang or GCC with NVPTX plugin)

### Build All Binaries

```bash
make
```

This compiles the following binaries and places them in the `bin/` directory:

#### OMPX-based collectives (using shared runtime):

* `bin/ompx_bcast`
* `bin/ompx_reduce`
* `bin/ompx_gather`
* `bin/ompx_scatter`
* `bin/ompx_allgather`
* `bin/ompx_allreduce`
* `bin/ompx_reduce_scatter`

#### Baseline OpenMP versions (naive implementations):

* `bin/omp_bcast`
* `bin/omp_reduce`
* `bin/omp_gather`
* `bin/omp_scatter`
* `bin/omp_allgather`
* `bin/omp_allreduce`
* `bin/omp_reduce_scatter`

### Clean Build Artifacts

```bash
make clean
```

Removes all compiled objects and the `bin/` directory.

## 🗂️ Project Structure

```
src/
├── ompx_collectives.[c|h]        # Shared OMPX Collectives runtime implementation
├── ompx_*.c                      # OMPX-based collective examples
├── omp_*.c                       # Naive OpenMP collective examples
```

## 🧪 Usage Example

### Broadcasting a Buffer to All Devices

To use the OMPX Collectives routines in your application, include the collective runtime header:

```c
#include "ompx_collectives.h"
```

Here's a minimal example using `ompx_target_broadcast` to broadcast a buffer from the host to multiple target devices:

```c
int NumDevices = omp_get_num_devices();
size_t SizePerDevice = ...; // buffer size in bytes
void *DevicesPtrs[NumDevices];
int DstDevices[NumDevices];

// Allocate buffers on each device
for (int i = 0; i < NumDevices; ++i) {
    DevicesPtrs[i] = omp_target_alloc(SizePerDevice, i);
    DstDevices[i] = i;
}

// Allocate and initialize a buffer on the host
int *HstPtr = malloc(SizePerDevice);
// Fill HstPtr with data...

int HstDevice = omp_get_initial_device();

// Perform the broadcast
ompx_target_broadcast(HstPtr, DevicesPtrs, SizePerDevice,
                      HstDevice, DstDevices, NumDevices);
```

### Running the Example App

To run the broadcast example:

```bash
./bin/ompx_bcast <buffer-size-in-bytes>
```

For example:

```bash
./bin/ompx_bcast 1048576
```

This will:

* Allocate a 1MB buffer on the host and on each target device
* Use `ompx_target_broadcast` to broadcast the host buffer to all devices
* Launch a parallel region where each device writes to its buffer
* Report the average runtime over multiple iterations

---

### 🚨 Notes

* You must have at least one OpenMP-compatible target device available (e.g., GPU).
* The collective routines assume the device pointers are allocated via `omp_target_alloc`.
* The host pointer must reside on `omp_get_initial_device()` (usually CPU).

See the `ompx_bcast` implementation in `src/ompx_bcast.c` for a complete reference.
