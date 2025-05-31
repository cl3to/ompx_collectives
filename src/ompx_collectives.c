#include <stdio.h>

#include "omp.h"
#include "ompx_collectives.h"

#define DEVICES_PER_WORKER 4

int ompx_get_num_devices_per_worker() {
    return DEVICES_PER_WORKER;
}

int ompx_get_num_head_devices() {
    return omp_get_num_devices()/ompx_get_num_devices_per_worker();
}

int ompx_is_head_device(int DeviceId) {
    return (DeviceId % DEVICES_PER_WORKER == 0);
}

int ompx_get_head_device(int DeviceId) {
    return (DeviceId/DEVICES_PER_WORKER) * DEVICES_PER_WORKER;
}

int ompx_get_relative_device_id(int DeviceId) {
    return DeviceId%DEVICES_PER_WORKER;
}

int ompx_get_head_id(int DeviceId) {
    return DeviceId/DEVICES_PER_WORKER;
}

void ompx_target_broadcast_naive(void *SrcPtr, void **DstPtrs, unsigned long Size,
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

void ompx_target_scatter_naive(void *SrcPtr, void **DstPtrs, unsigned long Count,
                         int SrcDevice, int *DstDevices, int NumDstDevices)
{
    int deps[NumDstDevices] = {};
    #pragma omp parallel num_threads(NumDstDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        void *DstDevicePtr = DstPtrs[DevIdx];
        unsigned long ChunkStart = DstDevice*Count;
        #pragma omp task depend(out: deps[DstDevice])
        {
            omp_target_memcpy(DstDevicePtr, SrcPtr, Count, 0, ChunkStart, DstDevice, SrcDevice);
        }        
    }
}

void ompx_target_gather_naive(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        int *SrcDevices, int DstDevice, int NumSrcDevices)
{
    int deps[NumSrcDevices] = {};
    #pragma omp parallel num_threads(NumSrcDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumSrcDevices; DevIdx++) {
        int SrcDevice = SrcDevices[DevIdx];
        void *SrcDevicePtr = SrcPtrs[DevIdx];
        unsigned long ChunkStart = SrcDevice*Count;

        if (DstDevice == SrcDevice)
            continue;

        #pragma omp task depend(out: deps[SrcDevice])
        {
            omp_target_memcpy(DstPtr, SrcDevicePtr, Count, ChunkStart, 0, DstDevice, SrcDevice);
        }        
    }
}

void ompx_target_broadcast(void *SrcPtr, void **DstPtrs, unsigned long Size,
                       int SrcDevice, int *DstDevices, int NumDstDevices)
{
    int deps[NumDstDevices] = {};
    int NumHeadDevices = ompx_get_num_head_devices();
    void *StagingPtrs[NumHeadDevices];

    #pragma omp parallel num_threads(NumDstDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        void *DstDevicePtr = DstPtrs[DevIdx];
        if (ompx_is_head_device(DstDevice)) {
            StagingPtrs[ompx_get_head_id(DstDevice)] = DstDevicePtr;
            #pragma omp task depend(out: deps[DstDevice])
            omp_target_memcpy(DstDevicePtr, SrcPtr, Size, 0, 0, DstDevice, SrcDevice);
        }

        else {
            int HeadDevice = ompx_get_head_device(DstDevice);
            void *HeadDevicePtr = StagingPtrs[ompx_get_head_id(HeadDevice)];
            #pragma omp task depend(in: deps[HeadDevice]) depend(out: deps[DstDevice])
            omp_target_memcpy(DstDevicePtr, HeadDevicePtr, Size, 0, 0, DstDevice, HeadDevice);
        }
    
    }
}

void ompx_target_scatter(void *SrcPtr, void **DstPtrs, unsigned long Count,
                         int SrcDevice, int *DstDevices, int NumDstDevices)
{
    int deps[NumDstDevices] = {};
    int NumHeadDevices = ompx_get_num_head_devices();
    void *StagingPtrs[NumHeadDevices];
    int HeadDevices[NumHeadDevices];
    size_t StagingSize = (Count * NumDstDevices) / NumHeadDevices;

    #pragma omp parallel for num_threads(NumHeadDevices)
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        if (ompx_is_head_device(DstDevice)) {
            int HeadId = ompx_get_head_id(DstDevice);
            StagingPtrs[HeadId] = omp_target_alloc(StagingSize, DstDevice);
            HeadDevices[HeadId] = DstDevice;
        }
    }

    #pragma omp parallel num_threads(NumDstDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumDstDevices; DevIdx++) {
        int DstDevice = DstDevices[DevIdx];
        void *DstDevicePtr = DstPtrs[DevIdx];
        unsigned long ChunkStart = DevIdx*Count;

        if (ompx_is_head_device(DstDevice)) {
            void *StagingPtr = StagingPtrs[ompx_get_head_id(DstDevice)];

            #pragma omp task depend(out: deps[DstDevice])
            omp_target_memcpy(StagingPtr, SrcPtr, StagingSize, 0, ChunkStart, DstDevice, SrcDevice);

            #pragma omp task depend(in: deps[DstDevice])
            omp_target_memcpy(DstDevicePtr, StagingPtr, Count, 0, 0, DstDevice, DstDevice);
        }

        else {
            int HeadDevice = ompx_get_head_device(DstDevice);
            void *StagingPtr = StagingPtrs[ompx_get_head_id(HeadDevice)];
            ChunkStart = ompx_get_relative_device_id(DstDevice)*Count;

            #pragma omp task depend(in: deps[HeadDevice]) depend(out: deps[DstDevice])
            omp_target_memcpy(DstDevicePtr, StagingPtr, Count, 0, ChunkStart, DstDevice, HeadDevice);
        }
    }
    
    #pragma omp parallel for num_threads(NumHeadDevices)
    for(int DevIdx = 0; DevIdx < NumHeadDevices; DevIdx++) {
        int HeadDevice = HeadDevices[DevIdx];
        omp_target_free(StagingPtrs[ompx_get_head_id(HeadDevice)], HeadDevice);
    }  
}

void ompx_target_gather(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        int *SrcDevices, int DstDevice, int NumSrcDevices)
{
    int deps[NumSrcDevices] = {};
    int NumHeadDevices = ompx_get_num_head_devices();
    void *StagingPtrs[NumHeadDevices];
    int HeadDevices[NumHeadDevices];
    unsigned long StagingSize = (Count*NumSrcDevices)/NumHeadDevices;

    #pragma omp parallel for num_threads(NumSrcDevices)
    for(int DevIdx = 0; DevIdx < NumSrcDevices; DevIdx++) {
        int SrcDevice = SrcDevices[DevIdx];

        if(ompx_is_head_device(SrcDevice)) {
            int HeadId = ompx_get_head_id(SrcDevice);
            if (SrcDevice == DstDevice)
                StagingPtrs[HeadId] = DstPtr;
            else
                StagingPtrs[HeadId] = omp_target_alloc(StagingSize, SrcDevice);
            HeadDevices[HeadId] = SrcDevice;
        }
    }

    #pragma omp parallel num_threads(NumSrcDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumSrcDevices; DevIdx++) {
        int SrcDevice = SrcDevices[DevIdx];
        void *SrcDevicePtr = SrcPtrs[DevIdx];

        int HeadDevice = ompx_get_head_device(SrcDevice);
        void *StagingPtr = StagingPtrs[ompx_get_head_id(HeadDevice)];

        int RelativeId = ompx_get_relative_device_id(SrcDevice);

        if (HeadDevice == DstDevice)
            RelativeId = SrcDevice;
        
        unsigned long ChunkStart = RelativeId*Count;

        #pragma omp task depend(out: deps[SrcDevice])
        omp_target_memcpy(StagingPtr, SrcDevicePtr, Count, ChunkStart, 0, HeadDevice, SrcDevice);
    }

    #pragma omp parallel num_threads(NumHeadDevices)
    #pragma omp single
    for(int DevIdx = 0; DevIdx < NumHeadDevices; DevIdx++) {
        int HeadDevice = HeadDevices[DevIdx];
        if (HeadDevice != DstDevice) {
            #pragma omp task
            {
                int RelativeId = ompx_get_head_id(HeadDevice);
                void *StagingPtr = StagingPtrs[RelativeId];
                unsigned long ChunkStart = RelativeId*StagingSize;
                omp_target_memcpy(DstPtr, StagingPtr, StagingSize, ChunkStart, 0, DstDevice, HeadDevice);
                omp_target_free(StagingPtr, HeadDevice);
            }
        }
    }
}

static inline void ompx_target_reduce_l(void *DstPtr, void *SrcPtr, void *ScratchPtr, unsigned long Count,
                                        unsigned TypeSize, ompx_reduce_op_t ROP, int DstDevice, int SrcDevice) {
    // 1) copy data from SrcPtr to ScrathPtr
    omp_target_memcpy(
        ScratchPtr, SrcPtr, Count*TypeSize, 0, 0, DstDevice, SrcDevice
    );

    // 2) Apply the reduce operation
    #pragma omp target teams device(DstDevice) \
            is_device_ptr(DstPtr, ScratchPtr) \
            firstprivate(Count, ROP)
    {
        ROP(ScratchPtr, DstPtr, Count);
    }
}

void ompx_target_reduce(void *DstPtr, void **SrcPtrs, unsigned long Count, unsigned TypeSize,
                        ompx_reduce_op_t Op, int *SrcDevices, int DstDevice,
                        int NumSrcDevices) {
    int Width = NumSrcDevices, Half = Width >> 1, NextWidth;
    unsigned long Size = Count * TypeSize;
    int Devs[NumSrcDevices];
    void *DevsPtrs[NumSrcDevices];
    void *ScratchPtrs[NumSrcDevices];

    for(int I = 0; I < NumSrcDevices; ++I) {
        Devs[I] = SrcDevices[I];
        DevsPtrs[I] = SrcPtrs[I];
    }
    
    // Need to allocate floor(Width/2) temp buffers to do the reduction
    for (int I = 0; I < Half; ++I)
        ScratchPtrs[I<<1] = omp_target_alloc(Size, I<<1);

    // Binary Tree-based reduction
    int level = 0;
    while(Width > 1) {
        Half = Width >> 1; // floor(Width/2)
        NextWidth = (Width + 1) >> 1; // ceil(Width/2)
        #pragma omp parallel num_threads(NumSrcDevices)
        #pragma omp single nowait
        for(int Idx = 0; Idx < Half; ++Idx) {
            // (dst=2*i, src=2*i+1)
            int DstDeviceIdx = (Idx<<1), SrcDeviceIdx = DstDeviceIdx+1;

            // use variables to store current values before create the task
            // this avoid race condition in Devs and DevsPtrs arrays access
            int DstDevice = Devs[DstDeviceIdx], SrcDevice = Devs[SrcDeviceIdx];
            void *DstPtr = DevsPtrs[DstDeviceIdx], *SrcPtr = DevsPtrs[SrcDeviceIdx];
            void *ScratchPtr = ScratchPtrs[Devs[DstDeviceIdx]];
            
            // Set the Idx that'll perform the redution
            // in the next level of the three (PIdx = Idx*2)
            Devs[Idx] = Devs[DstDeviceIdx];
            DevsPtrs[Idx] = DevsPtrs[DstDeviceIdx];
            
            // Apply the local reduction
            #pragma omp task
            ompx_target_reduce_l(
                DstPtr, SrcPtr, ScratchPtr,
                Count, TypeSize, Op, DstDevice, SrcDevice
            );

        }

        // If the Width is odd, put the non-reducted element
        // in the last possition on the next level of the three
        if (Width & 1) {
            Devs[NextWidth-1] = Devs[Width-1];
            DevsPtrs[NextWidth-1] = DevsPtrs[Width-1];
        }

        Width = NextWidth;
    }

    // Copy the Final Result to the DstDevice
    omp_target_memcpy(
        DstPtr, DevsPtrs[0], Size, 0, 0, DstDevice, Devs[0]
    );

    // Free Scratchpad
    for (int I = 0; I < Half; ++I)
        omp_target_free(ScratchPtrs[I<<1], I<<1);

}

void ompx_target_allreduce(void **DevPtrs, unsigned long Count, unsigned TypeSize,
    ompx_reduce_op_t Op, int *Devices, int NumDevices) {
    int Root = Devices[0];
    void *RootPtr = DevPtrs[0];
    // First execute the reduce to the root (DevPtrs[0])
    ompx_target_reduce(RootPtr, DevPtrs, Count, TypeSize, Op, Devices, Root, NumDevices);
    // Then Broascast that result to all the devices in the group
    ompx_target_broadcast(RootPtr, DevPtrs, Count*TypeSize, Root, Devices, NumDevices);
}

void ompx_target_reduce_scatter(void **DevPtrs, unsigned long Count, unsigned TypeSize,
                                ompx_reduce_op_t Op, int *Devices, int NumDevices) {
    int Root = Devices[0];
    void *RootPtr = DevPtrs[0];
    // First execute the reduce to the root (DevPtrs[0])
    ompx_target_reduce(RootPtr, DevPtrs, Count, TypeSize, Op, Devices, Root, NumDevices);
    // Then Scatter that result to all the devices in the group
    unsigned long SizePerDevice = Count*TypeSize / NumDevices;
    ompx_target_scatter(RootPtr, DevPtrs, SizePerDevice, Root, Devices, NumDevices);
}

void ompx_target_allgather(void **DevPtrs, unsigned long Count, int *Devices,
                           int NumDevices) {
    int Root = Devices[0];
    void *RootPtr = DevPtrs[0];
    // First Gather the data to the root (DevPtrs[0])
    ompx_target_gather_naive(RootPtr, DevPtrs, Count/NumDevices, Devices, Root, NumDevices);
    // Then Broadcast the full data to all devices
    ompx_target_broadcast(RootPtr, DevPtrs, Count, Root, Devices, NumDevices);
}