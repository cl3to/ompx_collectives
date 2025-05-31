#ifndef OMPX_COLLECTIVES_H
#define OMPX_COLLECTIVES_H

typedef void (*ompx_reduce_op_t) (
    void *InVec,
    void *InOutVec,
    unsigned long Count
);

void ompx_target_broadcast_naive(void *SrcPtr, void **DstPtrs,
                                 unsigned long Size, int SrcDevice,
                                 int *DstDevices, int NumDstDevices);

void ompx_target_scatter_naive(void *SrcPtr, void **DstPtrs,
                               unsigned long Count, int SrcDevice,
                               int *DstDevices, int NumDstDevices);

void ompx_target_gather_naive(void *DstPtr, void **SrcPtrs, unsigned long Count,
                              int *SrcDevices, int DstDevice,
                              int NumSrcDevices);

void ompx_target_broadcast(void *SrcPtr, void **DstPtrs, unsigned long Size,
                           int SrcDevice, int *DstDevices, int NumDstDevices);

void ompx_target_scatter(void *SrcPtr, void **DstPtrs, unsigned long Count,
                         int SrcDevice, int *DstDevices, int NumDstDevices);

void ompx_target_gather(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        int *SrcDevices, int DstDevice, int NumSrcDevices);

void ompx_target_reduce(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        unsigned TypeSize, ompx_reduce_op_t Op, int *SrcDevices,
                        int DstDevice, int NumSrcDevices);

void ompx_target_allreduce(void **DevPtrs, unsigned long Count, unsigned TypeSize,
                           ompx_reduce_op_t Op, int *Devices, int NumDevices);


void ompx_target_reduce_scatter(void **DevPtrs, unsigned long Count, unsigned TypeSize,
                                ompx_reduce_op_t Op, int *Devices, int NumDevices);

void ompx_target_allgather(void **DevPtrs, unsigned long Count, int *Devices,
                           int NumDevices);

#endif