#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"
#include "ompx_collectives.h"

#pragma omp begin declare target indirect
void ompx_reduce_sum_float(void *InVec, void *InOutVec, unsigned long Count)
{
    float *A = InVec;
    float *B = InOutVec;

    #pragma omp distribute parallel for
    for (unsigned long i = 0; i < Count; ++i) {
        B[i] += A[i];
    }
}
#pragma omp end declare target


int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("You must specify the per-device size of the buffer\n./ompx_allreduce <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t Length = SizePerDevice/sizeof(float);

    int HstDevice = omp_get_initial_device();
    void *DevicesPtrs[NumDevices];
    int Devices[NumDevices];
    double start, end;
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
        Devices[DevIdx] = DevIdx;
    }
    
    float *HstPtr = (float *)malloc(SizePerDevice);
    memset(HstPtr, 0, SizePerDevice);

    ompx_reduce_op_t ROP = ompx_reduce_sum_float;
    int warmups = 5;
    
    for(int I = 0; I < warmups; I++) {
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                float *DevicePtr = DevicesPtrs[DevIdx];
                #pragma omp target teams distribute parallel for\
                        is_device_ptr(DevicePtr)\
                        device(DevIdx) nowait
                {
                    for (unsigned long i = 0; i < Length; ++i) {
                        DevicePtr[i] = 1.0;
                    } 
                }
            }
        }

        ompx_target_reduce_scatter(DevicesPtrs, Length, sizeof(float), ROP, Devices, NumDevices);
    }

    int Samples = 5;
    double TotalTime = 0.0;

    for(int I = 0; I < Samples; ++I) {
        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                float *DevicePtr = DevicesPtrs[DevIdx];
                #pragma omp target teams distribute parallel for\
                        is_device_ptr(DevicePtr)\
                        device(DevIdx) nowait
                {
                    for (unsigned long i = 0; i < Length; ++i) {
                        DevicePtr[i] = 1.0;
                    } 
                }
            }
        }

        start = omp_get_wtime();
        ompx_target_reduce_scatter(DevicesPtrs, Length, sizeof(float), ROP, Devices, NumDevices);
        end = omp_get_wtime();

        TotalTime += (end-start);
    }

    omp_target_memcpy(HstPtr, DevicesPtrs[NumDevices-1], SizePerDevice, 0, 0, HstDevice, NumDevices-1);

    printf("R[0] = %lf\n", HstPtr[0]);
    printf("Runtime: %.4f seconds\n", TotalTime/Samples);

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
         omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

    return 0;
}
