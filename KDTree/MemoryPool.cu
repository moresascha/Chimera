#include "MemoryPool.h"
#include "../../Nutty/Nutty/cuda/cuda_helper.h"

__global__ void freeDeviceHeapMemory(void** toFree, size_t N)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N)
    {
        return;
    }

    void* ptr = toFree[id];
    if(ptr)
    {
        free(ptr);
    }

    toFree[id] = NULL;
}

__global__ void freeDeviceHeapMemory0(void* toFree)
{
    if(toFree)
    {
        free(toFree);
    }
}

__global__ void copyHeapToGlobal(void* heapMemory, void* globalMemory, size_t N)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N)
    {
        return;
    }
    byte c = ((byte*)heapMemory)[id];
    ((byte*)globalMemory)[id] = c;
}

extern "C"
{
    void FreeDeviceHeapMemory(void** toFree, size_t N)
    {
        size_t g = nutty::cuda::GetCudaGrid(N, (size_t)256);
        freeDeviceHeapMemory<<<g, 256>>>(toFree, N);
    }

    void CopyHeapToGlobal(void* heapMemory, void* globalMemory, size_t N)
    {
        size_t g = nutty::cuda::GetCudaGrid(N, (size_t)256);
        copyHeapToGlobal<<<g, 256>>>(heapMemory, globalMemory, N);
    }

    void FreeDeviceHeapMemory0(void* toFree)
    {
        freeDeviceHeapMemory0<<<1, 1>>>(toFree);
    }
}

cuDeviceHeap::cuDeviceHeap(void) : m_pDeviceMemoryPtrs(NULL), m_offset(0), m_nextOffset(0), m_size(0), m_pDeviceSizes(NULL)
{

}

cuDeviceHeap* cuDeviceHeap::GetDevPtr(void)
{
    return m_pDevPtr;
}

void cuDeviceHeap::Print(void)
{
    for(size_t i = 0; i < m_size; ++i)
    {
        size_t blockSize = GetBlockSize(i);
        void* ptr = GetBlockContent(i);
        DEBUG_OUT_A("Block 'id=%d' 'Bytes=%d' 'Address=%p'\n", i, blockSize, ptr);
    }
}

void cuDeviceHeap::Reset(void)
{
    FreeDeviceHeapMemory(m_pDeviceMemoryPtrs, m_size);
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceMemoryPtrs, 0, m_size * sizeof(void*)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceSizes, 0, m_size * sizeof(size_t)));
}

void cuDeviceHeap::Init(size_t initialSize)
{
    if(m_size)
    {
        return;
    }

    m_compactedSize = m_size = initialSize;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&m_pDeviceMemoryPtrs, m_size * sizeof(void*)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceMemoryPtrs, 0, m_size * sizeof(void*)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&m_pDeviceSizes, m_size * sizeof(size_t)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceSizes, 0, m_size * sizeof(size_t)));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&m_pDevPtr, sizeof(cuDeviceHeap)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_pDevPtr, this, sizeof(cuDeviceHeap), cudaMemcpyHostToDevice));
}

size_t cuDeviceHeap::GetBlockSize(size_t id)
{
    size_t size;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(&size, m_pDeviceSizes + id, sizeof(size_t), cudaMemcpyDeviceToHost));
    return size;
}

void* cuDeviceHeap::GetBlockContent(size_t id)
{
    void* _ptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(&_ptr, m_pDeviceMemoryPtrs + id, sizeof(void*), cudaMemcpyDeviceToHost));
    return _ptr;
}

size_t cuDeviceHeap::GetActiveBlocks(void)
{
    return m_compactedSize;
}

size_t cuDeviceHeap::GetSize(void)
{
    return m_size;
}

// void cuDeviceHeap::Prepare(size_t threadCount)
// {
//     m_offset = m_nextOffset;
//     m_nextOffset += threadCount;
// 
//     while(threadCount > (m_size + m_offset))
//     {
//         if(!Grow())
//         {
//             assert(0 && "Grow Failed.");
//         }
//     }
// }

void cuDeviceHeap::Free(void* ptr)
{
    FreeDeviceHeapMemory0(ptr);

    void** _ptr = (void**)malloc(sizeof(void*) * m_size);
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(&_ptr, m_pDeviceMemoryPtrs, sizeof(void*) * m_size, cudaMemcpyDeviceToHost));

    //slow
    for(size_t i = 0; i < m_size; ++i)
    {
        if(ptr == _ptr)
        {
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceMemoryPtrs + i, 0, sizeof(size_t)));
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(m_pDeviceSizes + i, 0, sizeof(size_t)));
        }
    }

    free(_ptr);
}

void cuDeviceHeap::Delete(void)
{
    if(m_pDeviceSizes)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFree(m_pDeviceSizes));
    }

    FreeDeviceHeapMemory(m_pDeviceMemoryPtrs, m_size);

    if(m_pDeviceMemoryPtrs)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFree(m_pDeviceMemoryPtrs));
    }
    m_pDeviceMemoryPtrs = NULL;
    m_pDeviceSizes = NULL;
}

bool cuDeviceHeap::Grow(void)
{
    void** ptrptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&ptrptr, 2 * m_size * sizeof(void*)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(ptrptr, m_pDeviceMemoryPtrs, m_size, cudaMemcpyDeviceToDevice));

    size_t* ptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&ptr, 2 * m_size * sizeof(size_t)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(ptr, m_pDeviceSizes, m_size, cudaMemcpyDeviceToDevice));

    Delete();

    m_size = 2 * m_size;

    m_pDeviceMemoryPtrs = ptrptr;
    m_pDeviceSizes = ptr;
    return true;
}

void cuDeviceHeap::Compact(void)
{
    size_t* sizes = (size_t*)malloc(m_size * sizeof(size_t));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(sizes, m_pDeviceSizes, m_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    void** content = (void**)malloc(m_size * sizeof(void*));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(content, m_pDeviceMemoryPtrs, m_size * sizeof(void*), cudaMemcpyDeviceToHost));

    size_t nextPos = 0;

    for(size_t i = 0; i < m_size; ++i)
    {
        if(content[i])
        {
            content[nextPos] = content[i];
            sizes[nextPos] = sizes[i];
            if(nextPos < i)
            {
                content[i] = NULL;
                sizes[i] = NULL;
            }
            nextPos++;
        }
    }

    m_compactedSize = nextPos;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_pDeviceSizes, sizes, m_size * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_pDeviceMemoryPtrs, content, m_size * sizeof(void*), cudaMemcpyHostToDevice));

    free(content);
    free(sizes);
}

cuDeviceHeap::~cuDeviceHeap(void)
{
    Delete();

    if(m_pDevPtr)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFree(m_pDevPtr));
    }
}

