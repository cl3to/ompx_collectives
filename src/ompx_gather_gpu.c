#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"
#include "ompx_collectives.h"

int main(int argc, char* argv[])
{

    if (argc != 2) {
        printf("You must specify the length of the buffer\n./ompx_gather <length>\n");
        return 1;
    }

    size_t Length = atol(argv[1]);
    size_t Size = Length*sizeof(int);
    int NumDevices = omp_get_num_devices();
    int HstDevice = omp_get_initial_device();
    void *DevicesPtrs[NumDevices];
    int Devices[NumDevices];
    double start, end;

    size_t Count = Size / NumDevices;
    size_t LengthPerDevice = Length / NumDevices;
    int DstDevice = 0;

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        if (DevIdx == DstDevice)
            DevicesPtrs[DevIdx] = omp_target_alloc(Size, DevIdx);
        else
            DevicesPtrs[DevIdx] = omp_target_alloc(Count, DevIdx);
        Devices[DevIdx] = DevIdx;
    }

    void *DstDevicePtr = DevicesPtrs[DstDevice];

    unsigned long DeviceSum = 0;
    int warmups = 1;

    start = omp_get_wtime();

    for(int I = 0; I < warmups; I++) {
        #pragma omp parallel num_threads(2)
        #pragma omp single
        {
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                void *DevicePtr = DevicesPtrs[DevIdx];
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                is_device_ptr(DevicePtr)\
                device(DevIdx)\
                nowait
                {
                    ((int*) DevicePtr)[0] = DevIdx;
                }
            }
        }
        
        ompx_target_gather_naive(DstDevicePtr, DevicesPtrs, Count, Devices, DstDevice, NumDevices);
    }

    end = omp_get_wtime();

    printf("Runtime: %.4f seconds\n", (end-start)/warmups);

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
         omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

    return 0;
}
