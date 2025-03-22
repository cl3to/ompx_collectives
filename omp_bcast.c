#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the length of the buffer\n./omp_bcast <length>\n");
        return 1;
    }

    size_t Length = atol(argv[1]);
    size_t Size = Length*sizeof(int);
    int NumDevices = omp_get_num_devices();
    int HstDevice = omp_get_initial_device();

    int *HstPtr = (int *)malloc(Size);
    unsigned long HstSum = 0;

    for(int I = 0; I < Length; I++) {
        HstPtr[I] = I;
        HstSum += HstPtr[I];
    }

    printf("Host: Sum = %lu\n", HstSum);

    #pragma omp parallel
    #pragma omp single
    {
        unsigned long DeviceSum = 0;
        int *DevicePtr = HstPtr;
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            #pragma omp target firstprivate(Length, DeviceSum, DevIdx)\
                        map(to: DevicePtr[:Length])\
                        device(DevIdx) nowait
            {
                #pragma omp for
                for(int I = 0; I < Length; I++) {
                    DeviceSum += DevicePtr[I];
                }
                printf("(Device=%d): Sum = %lu\n", DevIdx, DeviceSum);
            }
        }
    }


    free(HstPtr);

}