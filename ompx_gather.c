#include <stdio.h>
#include <stdlib.h>

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

    size_t Count = Size / NumDevices;
    size_t LengthPerDevice = Length / NumDevices;

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(Size, DevIdx);
        Devices[DevIdx] = DevIdx;
    }
    int *HstPtr = (int *)malloc(Size);

    #pragma omp parallel
    #pragma omp single
    {
        unsigned long DeviceSum = 0;
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            void *DevicePtr = DevicesPtrs[DevIdx];
            #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                        is_device_ptr(DevicePtr)\
                        device(DevIdx) nowait
            {
                #pragma omp for
                for(int I = 0; I < LengthPerDevice; I++) {
                    ((int*) DevicePtr)[I] = 1;
                    DeviceSum += ((int*) DevicePtr)[I];
                }
                printf("(Device=%d): Count = %lu\n", DevIdx, DeviceSum);
            }
        }
    }

    ompx_target_gather(HstPtr, DevicesPtrs, Count, Devices, HstDevice, NumDevices);

    unsigned long HstSum = 0;

    for(int I = 0; I < Length; I++) {
        HstSum += HstPtr[I];
    }

    printf("Host: Count = %lu\n", HstSum);

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
         omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

    return 0;
}
