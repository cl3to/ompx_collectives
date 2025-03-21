#include "omp.h"
#include "ompx_collectives.h"

void ompx_target_bcast(void *SrcPtr, void **DstPtrs, unsigned long Size,
                       int SrcDevice, int *DstDevices, int NumDstDevices)
{
    int deps[NumDstDevices] = {};
    #pragma omp parallel num_threads(NumDstDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        void *DstDevicePtr = DstPtrs[DevIdx];
        #pragma omp task depend(out: deps[DstDevice])
        {
            omp_target_memcpy(DstDevicePtr, SrcPtr, Size, 0, 0, DstDevice, SrcDevice);
        }
    }
}

void ompx_target_scatter(void *SrcPtr, void **DstPtrs, unsigned long Count,
                         int SrcDevice, int *DstDevices, int NumDstDevices)
{
    int deps[NumDstDevices] = {};
    #pragma omp parallel num_threads(NumDstDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        void *DstDevicePtr = DstPtrs[DevIdx];
        unsigned long ChunkStart = DevIdx*Count;
        #pragma omp task depend(out: deps[DstDevice])
        {
            omp_target_memcpy(DstDevicePtr, SrcPtr, Count, 0, ChunkStart, DstDevice, SrcDevice);
        }        
    }
}

void ompx_target_gather(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        int *SrcDevices, int DstDevice, int NumSrcDevices)
{
    int deps[NumSrcDevices] = {};
    #pragma omp parallel num_threads(NumSrcDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumSrcDevices; DevIdx++) {
        int SrcDevice = SrcDevices[DevIdx];
        void *SrcDevicePtr = SrcPtrs[DevIdx];
        unsigned long ChunkStart = DevIdx*Count;
        #pragma omp task depend(out: deps[SrcDevice])
        {
            omp_target_memcpy(DstPtr, SrcDevicePtr, Count, ChunkStart, 0, SrcDevice, DstDevice);
        }        
    }
}