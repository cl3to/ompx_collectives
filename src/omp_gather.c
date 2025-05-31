#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the per-device size of the buffer\n./omp_gather <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t Size = SizePerDevice*NumDevices;
    size_t LengthPerDevice = SizePerDevice/sizeof(int);
    
    int *HstPtr = (int *)malloc(Size);
    memset(HstPtr, 0, Size);
    
    double start, end;

    int warmups = 5;
    for(int I = 0; I < warmups; I++) {
        #pragma omp parallel num_threads(2)
        #pragma omp single
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                            map(from: DevicePtr[DevIdx*LengthPerDevice:LengthPerDevice])\
                            device(DevIdx) nowait
                {
                    DevicePtr[DevIdx*LengthPerDevice] = DevIdx;
                }
            }
        }
    }

    int Samples = 5;

    start = omp_get_wtime();
    for(int I=0; I<Samples; ++I) {
        #pragma omp parallel num_threads(2)
        #pragma omp single
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                            map(from: DevicePtr[DevIdx*LengthPerDevice:LengthPerDevice])\
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