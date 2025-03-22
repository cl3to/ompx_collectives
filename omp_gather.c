#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the length of the buffer\n./omp_gather <length>\n");
        return 1;
    }

    size_t Length = atol(argv[1]);
    size_t Size = Length*sizeof(int);
    int NumDevices = omp_get_num_devices();

    size_t LengthPerDevice = Length / NumDevices;

    int *HstPtr = (int *)malloc(Size);

    #pragma omp parallel
    #pragma omp single
    {
        unsigned long DeviceSum = 0;
        int *DevicePtr = HstPtr;
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            #pragma omp target firstprivate(LengthPerDevice, DeviceSum, DevIdx)\
                        map(from: DevicePtr[DevIdx*LengthPerDevice:LengthPerDevice])\
                        device(DevIdx) nowait
            {
                #pragma omp for
                for(int I = DevIdx*LengthPerDevice; I < DevIdx*LengthPerDevice+LengthPerDevice; I++) {
                    ((int*) DevicePtr)[I] = 1;
                    DeviceSum += ((int*) DevicePtr)[I];
                }
                printf("(Device=%d): Count = %lu\n", DevIdx, DeviceSum);
            }
        }
    }

    unsigned long HstSum = 0;

    for(int I = 0; I < Length; I++) {
        HstSum += HstPtr[I];
    }

    printf("Host: Count = %lu\n", HstSum);

    free(HstPtr);

}