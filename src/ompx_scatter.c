#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"
#include "ompx_collectives.h"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("You must specify the per-device size of the buffer\n./ompx_scatter <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t Size = SizePerDevice*NumDevices;
    size_t LengthPerDevice = SizePerDevice/sizeof(int);
    
    int HstDevice = omp_get_initial_device();
    void *DevicesPtrs[NumDevices];
    int DstDevices[NumDevices];
    double start, end;


    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
        DstDevices[DevIdx] = DevIdx;
    }

    int *HstPtr = (int *)malloc(Size);
    memset(HstPtr, 0, Size);
    
    int warmup = 5;
    for(int I = 0; I < warmup; I++) {
        ompx_target_scatter_naive(HstPtr, DevicesPtrs, SizePerDevice, HstDevice, DstDevices, NumDevices);
        
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                void *DevicePtr = DevicesPtrs[DevIdx];
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx) is_device_ptr(DevicePtr)\
                device(DevIdx) nowait
                {
                    ((int*)DevicePtr)[0] = DevIdx;
                }
            }
        }
    }

    int Samples = 5;

    start = omp_get_wtime();
    for(int I = 0; I < Samples; I++) {
        ompx_target_scatter_naive(HstPtr, DevicesPtrs, SizePerDevice, HstDevice, DstDevices, NumDevices);
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                void *DevicePtr = DevicesPtrs[DevIdx];
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx) is_device_ptr(DevicePtr)\
                device(DevIdx) nowait
                {
                    ((int*)DevicePtr)[0] = DevIdx;
                }
            }
        }
    }
    end = omp_get_wtime();

    printf("Runtime: %.4f seconds\n", (end-start)/Samples);

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
         omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

    return 0;
}
