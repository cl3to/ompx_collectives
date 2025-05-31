#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the per-device of the buffer\n./omp_bcast <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t LengthPerDevice = SizePerDevice/sizeof(int);

    int HstDevice = omp_get_initial_device();
    int *HstPtr = (int *)malloc(SizePerDevice);
    memset(HstPtr, 0, SizePerDevice);
    
    double start, end;

    int warmups = 5;
    for(int I = 0; I < warmups; I++) {
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                            map(to: DevicePtr[:LengthPerDevice])\
                            device(DevIdx) nowait
                {
                    DeviceSum += DevIdx;
                }
            }
        }
    }

    int Samples = 5;

    start = omp_get_wtime();
    for(int I=0; I<Samples; ++I){
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            unsigned long DeviceSum = 0;
            int *DevicePtr = HstPtr;
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                            map(to: DevicePtr[:LengthPerDevice])\
                            device(DevIdx) nowait
                {
                    DeviceSum += DevIdx;
                }
            }
        }
    }
    end = omp_get_wtime();

    printf("Runtime: %.4f seconds\n", (end-start)/Samples);

    free(HstPtr);

}