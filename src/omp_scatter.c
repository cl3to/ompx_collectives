#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the size of the per-device buffer\n./omp_scatter <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t Size = SizePerDevice*NumDevices;
    size_t LengthPerDevice = SizePerDevice/sizeof(int);
    
    double start, end;

    int *HstPtr = (int *)malloc(Size);
    memset(HstPtr, 0, Size);

    int Warmup = 5;

    for(int I = 0; I < Warmup; I++) {
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                        map(to: DevicePtr[DevIdx*LengthPerDevice:LengthPerDevice])\
                        device(DevIdx) nowait
                {
                    DevicePtr[DevIdx*LengthPerDevice] = DevIdx;
                }
            }
        }
    }

    int Samples = 5;

    start = omp_get_wtime();
    for(int I = 0; I < Samples; ++I) {
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                map(to: DevicePtr[DevIdx*LengthPerDevice:LengthPerDevice])\
                device(DevIdx) nowait
                {
                    DevicePtr[DevIdx*LengthPerDevice] = DevIdx;
                }
            }
        }
    }
    end = omp_get_wtime();

    printf("Runtime: %.4f seconds\n", (end-start)/Samples);

    free(HstPtr);
    return 0;
}