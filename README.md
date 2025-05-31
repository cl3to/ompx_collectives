# OMPX Collectives

This repository contains an experimental implementation of **OMPX Collectives**, an extension of OpenMP aimed at supporting high-level collective operations (e.g., broadcast, gather, reduce) in heterogeneous systems using target offloading.

The implementation focuses on enabling efficient collective operations across device and host threads, providing abstractions for data movement and synchronization in offloaded regions.

## Features

- 🔄 Collective operations for OpenMP target model
- 🚀 Optimized data movement for GPUs using topology-aware algorithms


## Motivation

Traditional OpenMP lacks high-level support for collective operations across devices. This project explores how such functionality can be added as a natural extension, facilitating the implementation of distributed parallel algorithms on offloaded regions.

## Usage

```cpp
int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("You must specify the per-device size of the buffer\n./ompx_bcast <size>\n");
        return 1;
    }


    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t LengthPerDevice = SizePerDevice/sizeof(int);

    int HstDevice = omp_get_initial_device();
    void *DevicesPtrs[NumDevices];
    int DstDevices[NumDevices];
    double start, end;

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
        DstDevices[DevIdx] = DevIdx;
    }

    int *HstPtr = (int *)malloc(SizePerDevice);
    memset(HstPtr, 1, SizePerDevice);

    ompx_target_broadcast(HstPtr, DevicesPtrs, SizePerDevice, HstDevice, DstDevices, NumDevices);

    #pragma omp parallel num_threads(2)
    #pragma omp single nowait
    {
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            void *DevicePtr = DevicesPtrs[DevIdx];
            #pragma omp target firstprivate(DevIdx) \
                    is_device_ptr(DevicePtr) device(DevIdx) nowait
            {
                ((int*)DevicePtr)[DevIdx] = DevIdx;
            }
        }
    }

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
         omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

    return 0;
}

```