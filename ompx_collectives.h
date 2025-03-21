#ifndef OMPX_COLLECTIVES_H
#define OMPX_COLLECTIVES_H

void ompx_target_bcast(void *SrcPtr, void **DstPtrs, unsigned long Size,
                       int SrcDevice, int *DstDevices, int NumDstDevices);

void ompx_target_scatter(void *SrcPtr, void **DstPtrs, unsigned long Count,
                         int SrcDevice, int *DstDevices, int NumDstDevices);

void ompx_target_gather(void *DstPtr, void **SrcPtrs, unsigned long Count,
                        int *SrcDevices, int DstDevice, int NumSrcDevices);

#endif