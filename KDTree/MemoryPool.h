#pragma once
#include "../../Nutty/Nutty/Nutty.h"
#include "../../Nutty/Nutty/Copy.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"
#include "../../Nutty/Nutty/HostBuffer.h"
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include "../Source/chimera/Logger.h"

extern "C"
{
    void FreeDeviceHeapMemory(void** toFree, size_t N);

    void FreeDeviceHeapMemory0(void* toFree);

    void CopyHeapToGlobal(void* heapMemory, void* globalMemory, size_t N);
};

class cuMemoryPool
{

};

class cuDeviceHeap
{

private:
    cuDeviceHeap* m_pDevPtr;
    void** m_pDeviceMemoryPtrs;
    size_t* m_pDeviceSizes;
    size_t m_size;
    size_t m_compactedSize;
    size_t m_offset;
    size_t m_nextOffset;

    __host__ bool Grow(void);

    __host__ void Delete(void);

public:
    __host__ cuDeviceHeap(void);

    __host__ size_t GetBlockSize(size_t id);

    __host__ void* GetBlockContent(size_t id);

    __host__ void Init(size_t initialSize);

    __host__ void Compact(void);

    __host__ cuDeviceHeap* GetDevPtr(void);

    template <typename T>
    __host__ void Copy(nutty::HostBuffer<T>& dst, size_t blockId);

    template <typename T>
    __host__ void Copy(nutty::DeviceBuffer<T>& dst, size_t blockId);

    //__host__ void Prepare(size_t threadCount);

    __host__ void Free(void* ptr);

    __host__ void Reset(void);

    __host__ size_t GetActiveBlocks(void);

    __host__ size_t GetSize(void);

    template <typename T>
    __device__ T* Alloc(unsigned int id, size_t size = 1);

    __device__ void* Alloc(unsigned int id, size_t size);

    __host__ void Print(void);

    __host__ ~cuDeviceHeap(void);
};

template <
    typename T
>
__host__ void cuDeviceHeap::Copy(nutty::HostBuffer<T>& dst, size_t blockId)
{
    size_t l = GetBlockSize(blockId);
    nutty::DeviceBuffer<T> _tmp(l/sizeof(T));
    CopyHeapToGlobal(GetBlockContent(blockId), (void*)_tmp.GetDevicePtr()(), l);
    nutty::Copy(dst.Begin(), _tmp.Begin(), _tmp.Size());
}

template <
    typename T
>
__host__ void cuDeviceHeap::Copy(nutty::DeviceBuffer<T>& dst, size_t blockId)
{
    size_t l = GetBlockSize(blockId);
    CopyHeapToGlobal(GetBlockContent(blockId), (void*)dst.GetDevicePtr()(), l);
}

template <typename T>
__device__ T* cuDeviceHeap::Alloc(unsigned int id, size_t size)
{
    return (T*)Alloc(id, sizeof(T) * size);
}

__forceinline __device__ void* cuDeviceHeap::Alloc(unsigned int id, size_t size)
{
    void* ptr = m_pDeviceMemoryPtrs[id];
    if(ptr)
    {
        return ptr;
    }

    ptr = malloc(size * sizeof(byte));
    m_pDeviceMemoryPtrs[id] = ptr;
    m_pDeviceSizes[id] = size;
    return ptr;
}