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
    void *StagingDevicesPtrs[NumDevices];
    
    float *HstPtr = (float *)malloc(SizePerDevice);
    memset(HstPtr, 0, SizePerDevice);
    
    int HstDevice = omp_get_initial_device();

    double start, end;

    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        DevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
        StagingDevicesPtrs[DevIdx] = omp_target_alloc(SizePerDevice, DevIdx);
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

        #pragma omp parallel num_threads(2)
        #pragma omp single nowait
        {
            float *DevicePtr = DevicesPtrs[0], *StagingPtr;
            for(int DevIdx = 1; DevIdx < NumDevices; DevIdx++) {
                DevicePtr = DevicesPtrs[DevIdx];
                StagingPtr = StagingDevicesPtrs[DevIdx];
                void *SrcDevicePtr = DevicesPtrs[DevIdx-1];
                omp_target_memcpy(StagingPtr, SrcDevicePtr, SizePerDevice, 0, 0, DevIdx, DevIdx-1);

                #pragma omp target teams device(DevIdx)\
                        is_device_ptr(DevicePtr, StagingPtr)
                {
                    #pragma omp distribute parallel for
                    for (unsigned long i = 0; i < Length; ++i) {
                        DevicePtr[i] += StagingPtr[i];
                    } 
                }
            }
            
            void *SrcPtr = DevicesPtrs[NumDevices-1];
            int SrcDevice = NumDevices-1;
            int Count = SizePerDevice/NumDevices;

            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                int DstDevice = DevIdx;
                void *DstDevicePtr = DevicesPtrs[DevIdx];
                unsigned long ChunkStart = DstDevice*Count;
                #pragma omp task
                {
                    omp_target_memcpy(DstDevicePtr, SrcPtr, Count, 0, ChunkStart, DstDevice, SrcDevice);
                }        
            }
        }

        omp_target_memcpy(HstPtr, DevicesPtrs[NumDevices-1], SizePerDevice, 0, 0, HstDevice, NumDevices-1);
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
        #pragma omp parallel num_threads(NumDevices)
        #pragma omp single nowait
        {
            float *DevicePtr = DevicesPtrs[0], *StagingPtr;
            for(int DevIdx = 1; DevIdx < NumDevices; DevIdx++) {
                DevicePtr = DevicesPtrs[DevIdx];
                StagingPtr = StagingDevicesPtrs[DevIdx];
                void *SrcDevicePtr = DevicesPtrs[DevIdx-1];
                omp_target_memcpy(StagingPtr, SrcDevicePtr, SizePerDevice, 0, 0, DevIdx, DevIdx-1);

                #pragma omp target teams device(DevIdx)\
                        is_device_ptr(DevicePtr, StagingPtr)
                {
                    #pragma omp distribute parallel for
                    for (unsigned long i = 0; i < Length; ++i) {
                        DevicePtr[i] += StagingPtr[i];
                    } 
                }
            }

            void *SrcPtr = DevicesPtrs[NumDevices-1];
            int SrcDevice = NumDevices-1;
            int Count = SizePerDevice/NumDevices;

            for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
                int DstDevice = DevIdx;
                void *DstDevicePtr = DevicesPtrs[DevIdx];
                unsigned long ChunkStart = DstDevice*Count;
                #pragma omp task
                {
                    omp_target_memcpy(DstDevicePtr, SrcPtr, Count, 0, ChunkStart, DstDevice, SrcDevice);
                }        
            }
        }
        end = omp_get_wtime();
        
        TotalTime += (end-start);
    }
    
    omp_target_memcpy(HstPtr, DevicesPtrs[NumDevices-1], SizePerDevice, 0, 0, HstDevice, NumDevices-1);

    printf("R[0] = %lf\n", HstPtr[0]);
    printf("Runtime: %.4f seconds\n", TotalTime/Samples);

    free(HstPtr);
    for(int DevIdx = 0; DevIdx < NumDevices; DevIdx++) {
        omp_target_free(DevicesPtrs[DevIdx], DevIdx);
        omp_target_free(StagingDevicesPtrs[DevIdx], DevIdx);
    }

   return 0;
}