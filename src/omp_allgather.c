#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omp.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("You must specify the per-device size of the buffer\n./omp_reduce <size>\n");
        return 1;
    }

    size_t SizePerDevice = atol(argv[1]);
    int NumDevices = omp_get_num_devices();
    size_t Length = SizePerDevice/sizeof(float);
    
    void *DevicesPtrs[NumDevices];
    
    float *HstPtr = (float *)malloc(SizePerDevice);
    memset(HstPtr, 0, SizePerDevice);
    
    int HstDevice = omp_get_initial_device();

    double start, end;

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
    }

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

        void *RootPtr = DevicesPtrs[0];
        int Count = SizePerDevice/NumDevices;

        #pragma omp parallel for num_threads(NumDevices)
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            int SrcDevice = DevIdx;
            void *SrcDevicePtr = DevicesPtrs[DevIdx];
            unsigned long ChunkStart = SrcDevice*Count;
            omp_target_memcpy(RootPtr, SrcDevicePtr, Count, ChunkStart, 0, 0, SrcDevice);
        }

        #pragma omp parallel for num_threads(NumDevices)
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            void *DPtr = DevicesPtrs[DevIdx];
            omp_target_memcpy(DPtr, RootPtr, SizePerDevice, 0, 0, DevIdx, 0);
        }

        omp_target_memcpy(HstPtr, DevicesPtrs[0], SizePerDevice, 0, 0, HstDevice, 0);
    }

    int Samples = 5;
    double TotalTime = 0.0;
    
    for(int I=0; I<Samples; I++) {
        memset(HstPtr, 0, SizePerDevice);

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

        void *RootPtr = DevicesPtrs[0];
        int Count = SizePerDevice/NumDevices;

        #pragma omp parallel for num_threads(NumDevices)
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            int SrcDevice = DevIdx;
            void *SrcDevicePtr = DevicesPtrs[DevIdx];
            unsigned long ChunkStart = SrcDevice*Count;
            omp_target_memcpy(RootPtr, SrcDevicePtr, Count, ChunkStart, 0, 0, SrcDevice);
        }

        #pragma omp parallel for num_threads(NumDevices)
        for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
            void *DPtr = DevicesPtrs[DevIdx];
            omp_target_memcpy(DPtr, RootPtr, SizePerDevice, 0, 0, DevIdx, 0);
        }

        end = omp_get_wtime();
        
        TotalTime += (end-start);
    }
    
    omp_target_memcpy(HstPtr, DevicesPtrs[0], SizePerDevice, 0, 0, HstDevice, 0);

    printf("R[0] = %lf\n", HstPtr[0]);
    printf("Runtime: %.4f seconds\n", TotalTime/Samples);

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        omp_target_free(DevicesPtrs[DevIdx], DevIdx);
    }

   return 0;
}