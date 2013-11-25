#include "kdtree.cuh"

#include <windows.h>
#include "../../Nutty/Nutty/Nutty.h"
#include "../../Nutty/Nutty/Fill.h"
#include "../../Nutty/Nutty/ForEach.h"
#include "../../Nutty/Nutty/Reduce.h"
#include "../../Nutty/Nutty/Functions.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"
#include "../../Nutty/Nutty/cuda/Module.h"
#include "../../Nutty/Nutty/cuda/Kernel.h"
#include "../../Nutty/Nutty/Sort.h"
#include "../../Nutty/Nutty/Wrap.h"
#include "../../Nutty/Nutty/Scan.h"

#include "../Source/chimera/Logger.h"

#include <sstream>

template<>
struct ShrdMemory<Split>
{
    __device__ Split* Ptr(void) 
    { 
        extern __device__ __shared__ Split s_split[];
        return s_split;
    }
};

bool debug_out = false; 
 
void print(const char* str)
{
    OutputDebugStringA(str);
}

struct SAH
{
    __device__ Split operator()(Split t0, Split t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

struct float3min
{
    __device__ float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Min<float>()(t0.x, t1.x);
        r.y = nutty::binary::Min<float>()(t0.y, t1.y);
        r.z = nutty::binary::Min<float>()(t0.z, t1.z);
        return r;
    }
};

struct float3max
{
    __device__  float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Max<float>()(t0.x, t1.x);
        r.y = nutty::binary::Max<float>()(t0.y, t1.y);
        r.z = nutty::binary::Max<float>()(t0.z, t1.z);
        return r;
    }
};

void printi(int i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    print(ss.str().c_str());
}

template < typename T>
struct BufferPrinter
{
    uint m_c;
    uint m_r;
    BufferPrinter(uint r = (uint)-1) : m_c(0), m_r(r)
    {

    }

    void operator()(T i)
    {
        printi(i);
        m_c++;
        if(m_c % m_r == 0)
        {
            print("\n");
        }
    }
};

void printsplit(Split i)
{
    std::stringstream ss;
    ss << "(axis=";
    ss << i.axis;
    ss << ", split=";
    ss << i.split;
    ss << ", sah=";
    ss << i.sah;
    ss << ", below=";
    ss << i.below;
    ss << ", above=";
    ss << i.above;
    ss << ")";
    print(ss.str().c_str());
}

void printf3(float3 i)
{
    std::stringstream ss;
    ss << "(x=";
    ss << i.x;
    ss << ", y=";
    ss << i.y;
    ss << ", z=";
    ss << i.z;
    ss << ") ";
    print(ss.str().c_str());
}

void printFloat(float i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    print(ss.str().c_str());
}

void printFloat2(float2 i)
{
    std::stringstream ss;
    ss << "v=" << i.x;
    ss << " ";
    ss << "split=" << i.y;
    ss << ", ";
    print(ss.str().c_str());
}

void printBuffer(nutty::DeviceBuffer<uint>& b, uint r = (uint)-1)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), BufferPrinter<uint>(r));
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<Split>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printsplit);
    print("\n");
}

template < typename T>
void printBuffer(T& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printf3);
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<float>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printFloat);
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<float2>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printFloat2);
    print("\n");
}

void printContentCount(nutty::DeviceBuffer<uint>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printi);
    print("\n");
}

template <
    typename T
>
struct clearT
{
    T operator()(void) { return T(); }
};

struct clearuint
{
    uint m_v;
    clearuint(uint v) : m_v(v)
    {

    }
    uint operator()(void) { return m_v; }
};

struct AxisSort
{
    char axis;
    AxisSort(char a) : axis(a)
    {

    }
    __device__ __host__ char operator()(float3 f0, float3 f1)
    {
        return getAxis(&f0, axis) > getAxis(&f1, axis);
    }
};

__global__ void setNodesCount(uint* prefixNodesContentCount, uint* nodesContentCount, Split* allSplits, Split* nodeSplits, uint count, uint depth)
{
    uint id = GlobalId;
//     if(id >= (1 << depth))
//     {
//         return;
//     }
    uint contentCountOffset = elemsBeforeNextLevel(depth);//(1 << (depth+1)) - 1;
	uint contentCountOffsetMe = elemsBeforeLevel(depth); //(1 << depth) - 1;
    uint sa = 0;
    int last = -1;
    for(uint i = 0; i < id; ++i)
    {
        last = nodesContentCount[contentCountOffsetMe + i];
        sa += last;
    }

    Split s = allSplits[sa];
    if(last > 0 || sa == 0)
    {
        nodesContentCount[contentCountOffset + 2 * id + 0] = s.below;
        nodesContentCount[contentCountOffset + 2 * id + 1] = s.above;
    }
 	nodeSplits[contentCountOffsetMe + id] = s;
}

template <
    typename Data,
    typename Index
>
void printindexcontent(Data& d, Index& index, size_t size)
{
    if(!debug_out)
    {
        return;
    }
    nutty::HostBuffer<float3> t(1);
    for(int i = 0; i < size; ++i)
    {
        auto it = index + i;
        uint id = *it;
        nutty::Copy(t.Begin(), d + id, 1);
        DEBUG_OUT_A("[%f] %f %f, ", t[0].x, t[0].y, t[0].z);
    }
    DEBUG_OUT("\n");
}

template <
    typename T
>
class kdTree : public IKDTree
{
public:
    void* m_pBuffer[eBufferCount];
    void* m_pClearBuffer[eBufferCount];
    byte m_depth;
    byte m_maxDepth;
    uint m_nodesCount;
    uint m_elements;
    nutty::MappedPtr<T>* m_data;
    nutty::cuModule* m_cudaModule;
    nutty::cuKernel* m_computePossibleSplits;
    nutty::cuKernel* m_scan;
    nutty::cuKernel* m_spread;
    nutty::cuKernel* m_splitBBox;

    nutty::HostBuffer<Split> splitTmp;
    nutty::HostBuffer<Split> splitDataTmp;
    nutty::HostBuffer<uint> hostContentCount;
    nutty::HostBuffer<uint> countTmp;
    nutty::HostBuffer<float3> bboxTmp;

    nutty::DeviceBuffer<uint> m_perThreadNodePos;
	nutty::DeviceBuffer<uint> m_prefixSumNodeCount;

    nutty::DeviceBuffer<float3> m_aabbMin;
    nutty::DeviceBuffer<float3> m_aabbMax;

    nutty::DeviceBuffer<uint> m_null;

    kdTree(byte depth, byte maxDepth) : m_cudaModule(NULL), m_depth(depth), m_maxDepth(maxDepth), m_nodesCount(0), m_computePossibleSplits(NULL), m_scan(NULL), m_elements(0)
    {    
        for(int i = 0; i < eBufferCount; ++i)
        {
            m_pBuffer[i] = m_pClearBuffer[i] = NULL;
        }
        countTmp.Resize((1 << (maxDepth)));
        bboxTmp.Resize((1 << (maxDepth)));
    }

    std::stringstream _stream;
    void GetContentCountStr(std::string& s)
    {
        nutty::Copy(hostContentCount.Begin(), GetBuffer<uint>(eNodesContentCount).Begin(), m_nodesCount);
        s.clear();
        _stream.str("");
        for(uint i = 0; i < (1<<m_depth)-1; ++i)
        {
            _stream << hostContentCount[i] << " ";
        }

        s = _stream.str();
    }

    template <
        typename B
    >
    nutty::DeviceBuffer<B>& GetBuffer(DeviceBufferType type)
    {
        assert(m_pBuffer[type] != NULL);
        return *((nutty::DeviceBuffer<B>*)m_pBuffer[type]);
    }
    
    uint GetCurrentDepth(void)
    {
        return m_depth;
    }

    void SetDepth(uint d)
    {
        m_depth = min(m_maxDepth, max(1, d));
    }

    ~kdTree(void)
    {
        for(int i = 0; i < eBufferCount; ++i)
        {
            SAFE_DELETE(m_pBuffer[i]);
            SAFE_DELETE(m_pClearBuffer[i]);
        }
        SAFE_DELETE(m_computePossibleSplits);
        SAFE_DELETE(m_cudaModule);
        SAFE_DELETE(m_scan);
        SAFE_DELETE(m_spread);
        SAFE_DELETE(m_splitBBox);
    }

    void Init(void* data, uint elements)
    {
        m_data = (nutty::MappedPtr<T>*)data;

        m_elements = elements;

        splitTmp.Resize(elements);

        m_nodesCount = (1 << (uint)m_maxDepth);

        hostContentCount.Resize(m_nodesCount);
        
        splitDataTmp.Resize((m_nodesCount+1)/2);

        m_aabbMin.Resize(m_elements);
        m_aabbMax.Resize(m_elements);

        InitBuffer();

        m_cudaModule = new nutty::cuModule("./ptx/KD.ptx");

        m_computePossibleSplits = new nutty::cuKernel(m_cudaModule->GetFunction("computeSplits"));
        uint block = elements < 256 ? elements : 256;
        uint grid = max(1, (uint)elements / block);
        m_computePossibleSplits->SetDimension(grid, block);
        m_computePossibleSplits->SetKernelArg(0, GetBuffer<float3>(eAxisAlignedBB));
        m_computePossibleSplits->SetKernelArg(1, GetBuffer<uint>(eNodesContent));
        m_computePossibleSplits->SetKernelArg(2, GetBuffer<uint>(eNodesContentCount));
        m_computePossibleSplits->SetKernelArg(3, GetBuffer<float2>(ePosSplits));
        m_computePossibleSplits->SetKernelArg(7, m_perThreadNodePos);
        m_computePossibleSplits->SetKernelArg(8, m_prefixSumNodeCount);
        m_computePossibleSplits->SetKernelArg(9, GetBuffer<uint>(eSplitData));
            

        m_scan = new nutty::cuKernel(m_cudaModule->GetFunction("scanKD"));
        m_scan->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_scan->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_scan->SetSharedMemory((uint)elements * 2 * sizeof(uint));

        m_spread = new nutty::cuKernel(m_cudaModule->GetFunction("spreadContent"));
        m_spread->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_spread->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_spread->SetKernelArg(2, GetBuffer<uint>(eSplitData));
        m_spread->SetDimension(grid, block);

        m_splitBBox = new nutty::cuKernel(m_cudaModule->GetFunction("splitNodes"));

        m_splitBBox->SetKernelArg(0, GetBuffer<uint>(eAxisAlignedBB));
        m_splitBBox->SetKernelArg(1, GetBuffer<uint>(eSplitData));
    }

    void Generate(void)
    {
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
        nutty::DeviceBuffer<Split>& posSplits = GetBuffer<Split>(ePosSplits); //SAHSplit
        nutty::DeviceBuffer<Split>& splitData = GetBuffer<Split>(eSplitData);
        
        size_t elementCount = m_elements;

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;
        
        uint stride = (uint)elementCount;

        nutty::DevicePtr<T> dataBuffer = m_data->Bind();

        m_computePossibleSplits->SetKernelArg(4, dataBuffer);
        m_spread->SetKernelArg(3, dataBuffer);

//          nutty::base::ReduceIndexed(aabbs.Begin(), dataBuffer, dataBuffer + elementCount, nodesContent.Begin(), max3f, float3min());
//          nutty::base::ReduceIndexed(aabbs.Begin()+1, dataBuffer, dataBuffer + elementCount, nodesContent.Begin(), min3f, float3max());

        nutty::Reduce(m_aabbMin.Begin(), dataBuffer, dataBuffer + elementCount, float3min());
        nutty::Reduce(m_aabbMax.Begin(), dataBuffer, dataBuffer + elementCount, float3max());

        nutty::Copy(aabbs.Begin(), m_aabbMin.Begin(), 1);
        nutty::Copy(aabbs.Begin()+1, m_aabbMax.Begin(), 1);

        for(int i = 0; i < m_depth-1; ++i)
        {
            /*
            if(i > 0)
            {
                uint invalidAddress = (uint)-1;
                
//                 printBuffer(nodesContent);
//                 printBuffer(nodesContentCount);
// 
//                 for(int k = 0; k < (1 << i); ++k)
//                 {
//                     uint nodesContenCounttOffset = ((1 << i) - 1) + k;
//                     uint nodesContentOffset = offset + k * elementCount;
//                     if(debug_out)
//                     {
//                         DEBUG_OUT_A("%d %d\n", nodesContenCounttOffset, nodesContentOffset);
//                     }
//                     auto contentStart = nodesContent.Begin() + nodesContentOffset;
//                     auto countStart = nodesContentCount.Begin() + nodesContenCounttOffset;
//                     auto scanTmpStart = m_scanTmp.Begin() + nodesContentOffset;
//                     nutty::Scan(contentStart, contentStart + elementCount, scanTmpStart, countStart);
//                 }
				printBuffer(nodesContentCount);
                printBuffer(nodesContent, elementCount);
                auto contentStart = nodesContent.Begin() + elementCount;
                auto countStart = nodesContentCount.Begin() + 1;
                auto scanTmpStart = m_scanTmp.Begin();
                nutty::Scan(contentStart, contentStart + elementCount, scanTmpStart, countStart);

                contentStart = nodesContent.Begin() + 2 * elementCount;
                countStart = nodesContentCount.Begin() + 2;
                scanTmpStart = m_scanTmp.Begin() + elementCount;
                nutty::Scan(contentStart, contentStart + elementCount, scanTmpStart, countStart);
                
//                 m_scan->SetDimension(1 << (i), (uint)elementCount / 2);
//                 m_scan->SetKernelArg(2, invalidAddress);
//                 m_scan->SetKernelArg(3, offset);
//                 m_scan->SetKernelArg(4, i);
//                 m_scan->Call();
				printBuffer(m_scanTmp);
                printBuffer(nodesContentCount);
                printBuffer(nodesContent, elementCount);

                offset += elementCount * (1 << (i));
            }
            */
            /*for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                auto start0 = aabbs.Begin() + (size_t)((2*(j-1)));
                auto start1 = aabbs.Begin() + (size_t)(2*j-1);
                
                nutty::base::ReduceIndexed(start0, dataBuffer, dataBuffer + elementCount, nodesContent.Begin() + ((j-1) * elementCount), max3f, float3min());
                nutty::base::ReduceIndexed(start1, dataBuffer, dataBuffer + elementCount, nodesContent.Begin() + ((j-1) * elementCount), min3f, float3max());
            }*/
// 
//             printBuffer(nodesContentCount);
//             printBuffer(nodesContent, elementCount);

            
            uint contentSum = 0;
	
            uint copyStartAdd = (1 << i) - 1;
            uint copyLength = ((1 << (i+1)) - 1) - copyStartAdd;

            nutty::Copy(bboxTmp.Begin() + 2 * copyStartAdd, aabbs.Begin() + 2 * copyStartAdd, copyLength * 2);

//            printBuffer(bboxTmp);

            nutty::Copy(countTmp.Begin() + copyStartAdd, nodesContentCount.Begin() + copyStartAdd, copyLength);

/*            nutty::Copy(splitDataTmp.Begin() + copyStartAdd, splitData.Begin() + copyStartAdd, copyLength);*/
//             printBuffer(nodesContent);
//             printindexcontent(dataBuffer, nodesContent.Begin(), elementCount);
            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                float3 min = bboxTmp[(size_t)(2*(j-1))];
                float3 max = bboxTmp[(size_t)(2*j-1)];
                
                int axis = getLongestAxis(min, max);

                auto begin = nodesContent.Begin() + contentSum;

                uint c = countTmp[(size_t)(j-1)];
                if(c)
                {
                   // DEBUG_OUT_A("%d %d\n", contentSum, nutty::Distance(begin, begin + c));
					nutty::Sort(begin, begin + c, dataBuffer, AxisSort(axis));
					//printBuffer(nodesContent);
                    //printindexcontent(dataBuffer, begin, c);
                }
                contentSum += c;
            }
		
            //printBuffer(nodesContent);
            //nutty::Copy(splitData.Begin() + copyStartAdd, splitDataTmp.Begin() + copyStartAdd, copyLength);

            m_computePossibleSplits->SetKernelArg(5, elementCount);
            m_computePossibleSplits->SetKernelArg(6, i);
            m_computePossibleSplits->Call();
            printBuffer(posSplits);
 
            contentSum = 0;

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                int c = (int)countTmp[(size_t)(j-1)];
                if((int)(c) > 0)
                {
                    auto begin = posSplits.Begin() + contentSum;
                    //DEBUG_OUT_A("%d %d %d %d\n", j, i, c, contentSum);
                    if(c > 1)
                    {
                        nutty::Reduce(begin, begin + c, SAH());
                    }
//                     printsplit(*begin);
//                     DEBUG_OUT("\n");
                }
                //if((int)(c) > 0)
                {
                    contentSum += c;
                }
            }
            uint g = 1;
            uint b = (1 << i);
            if(b > 256)
            {
                g = nutty::cuda::getCudaGrid(b, 256U);
                b = 256;
            }
            //printBuffer(posSplits);

            setNodesCount<<<g, b>>>(m_prefixSumNodeCount.Begin()(), nodesContentCount.Begin()(), posSplits.Begin()(), splitData.Begin()(), m_elements, i);
//             printBuffer(nodesContentCount);
//             printBuffer(splitData);
			/*
            nutty::DeviceBuffer<uint> sums(1);

            uint b = ((1 << (i+1)) - 1);
            uint e = (1 << (i+1));
            auto scanBegin = nodesContentCount.Begin() + b;
            nutty::PrefixSumScan(scanBegin, scanBegin + e, m_prefixSumNodeCount.Begin() + b, sums.Begin());
            *(
            /*
            nutty::Copy(splitTmp.Begin(), posSplits.Begin(), m_elements);

            contentSum = 0;

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                uint c = countTmp[(size_t)(j-1)];
                SplitData sp = splitDataTmp[(size_t)(j-1)];
                if(c)
                {
                    //printBuffer(posSplits);

                    float2 s = splitTmp[contentSum];
                    sp.split = s.y;
                    splitDataTmp.Insert(j-1, sp);
                }
                else
                {
                    sp.split = FLT_MAX;
                    splitDataTmp.Insert(j-1, sp);
                }
                contentSum += c;
            }

            nutty::Copy(splitData.Begin() + copyStartAdd, splitDataTmp.Begin() + copyStartAdd, copyLength); */

//             if(i+1 == m_depth-1)
//             {
//                 break;
//             }

            m_spread->SetKernelArg(4, stride);
            m_spread->SetKernelArg(5, i);
            m_spread->SetKernelArg(6, m_perThreadNodePos);
            m_spread->SetKernelArg(7, m_prefixSumNodeCount);
            m_spread->Call();

			printBuffer(m_prefixSumNodeCount, elementCount);
            printBuffer(m_perThreadNodePos, elementCount);
// 			printBuffer(nodesContent, elementCount);

            printBuffer(nodesContentCount);

            m_splitBBox->SetDimension(g, b);
            m_splitBBox->SetKernelArg(2, i);
            m_splitBBox->Call();
        }
        cuCtxSynchronize();
        m_data->Unbind();

        //printContentCount(nodesContentCount);
    }

    void Update(void)
    {
        ClearBuffer();
        Generate();
    }

    void* GetBuffer(DeviceBufferType b)
    {
        return m_pBuffer[b];
    }

private:
    void InitBuffer(void)
    {
        size_t elementCount = m_elements;

        float3 init = float3();
        nutty::DeviceBuffer<float3>* aabbs = new nutty::DeviceBuffer<float3>(2 * m_nodesCount, init);
        m_pBuffer[eAxisAlignedBB] = aabbs;
        m_pClearBuffer[eAxisAlignedBB] = new nutty::DeviceBuffer<float3>(2 * m_nodesCount, init);

        nutty::DeviceBuffer<uint>* nodesContentCount = new nutty::DeviceBuffer<uint>(m_nodesCount, 0);
        m_pBuffer[eNodesContentCount] = nodesContentCount;
        nodesContentCount->Insert(0, (uint)elementCount); //first node has all content in it

        nodesContentCount = new nutty::DeviceBuffer<uint>(m_nodesCount, 0);
        nodesContentCount->Insert(0, (uint)elementCount); //first node has all content in it
        m_pClearBuffer[eNodesContentCount] = nodesContentCount;

        m_prefixSumNodeCount.Resize(elementCount);
        m_perThreadNodePos.Resize(elementCount);
        m_null.Resize(elementCount);

        nutty::HostBuffer<uint> null(m_prefixSumNodeCount.Size(), 0);

        nutty::Copy(m_null.Begin(), null.Begin(), null.Size());
        nutty::Copy(m_prefixSumNodeCount.Begin(), m_null.Begin(), m_prefixSumNodeCount.Size());
        nutty::Copy(m_perThreadNodePos.Begin(), m_null.Begin(), m_perThreadNodePos.Size());

        nutty::DeviceBuffer<uint>* nodesContent = new nutty::DeviceBuffer<uint>(elementCount, 0xFFFFFFFF);
        m_pBuffer[eNodesContent] = nodesContent;
        nutty::Fill(nodesContent->Begin(), nodesContent->Begin() + elementCount, nutty::unary::Sequence<uint>());

        nodesContent = new nutty::DeviceBuffer<uint>(elementCount, 0xFFFFFFFF);
        m_pClearBuffer[eNodesContent] = nodesContent;
        nutty::Fill(nodesContent->Begin(), nodesContent->Begin() + elementCount, nutty::unary::Sequence<uint>());

        Split s;
        ZeroMemory(&s, sizeof(Split)); 
        s.sah = FLT_MAX;

        nutty::DeviceBuffer<Split>* possibleSplits = new nutty::DeviceBuffer<Split>(elementCount, s);
        m_pBuffer[ePosSplits] = possibleSplits;

        possibleSplits = new nutty::DeviceBuffer<Split>(elementCount, s);
        m_pClearBuffer[ePosSplits] = possibleSplits;

        nutty::DeviceBuffer<Split>* splitData = new nutty::DeviceBuffer<Split>((m_nodesCount+1)/2, s);
        m_pBuffer[eSplitData] = splitData;

        splitData = new nutty::DeviceBuffer<Split>((m_nodesCount+1)/2, s);
        m_pClearBuffer[eSplitData] = splitData;
    }

public:
    void ClearBuffer(void)
    {
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<float3>* _clearaabbs = (nutty::DeviceBuffer<float3>*)m_pClearBuffer[eAxisAlignedBB];
        nutty::Copy(aabbs.Begin(), _clearaabbs->Begin(), aabbs.Size());

        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>* _clearNodes = (nutty::DeviceBuffer<uint>*)m_pClearBuffer[eNodesContent];
        nutty::Copy(nodesContent.Begin(), _clearNodes->Begin(), nodesContent.Size());

        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
        nutty::DeviceBuffer<uint>* _clearNodesC = (nutty::DeviceBuffer<uint>*)m_pClearBuffer[eNodesContentCount];
        nutty::Copy(nodesContentCount.Begin(), _clearNodesC->Begin(), nodesContentCount.Size());

        nutty::DeviceBuffer<Split>& splits = GetBuffer<Split>(eSplitData);
        nutty::DeviceBuffer<Split>* _clearSplit = (nutty::DeviceBuffer<Split>*)m_pClearBuffer[eSplitData];
        nutty::Copy(splits.Begin(), _clearSplit->Begin(), splits.Size());

        nutty::DeviceBuffer<Split>* _clearPosSplit = (nutty::DeviceBuffer<Split>*)m_pClearBuffer[ePosSplits];
        nutty::DeviceBuffer<Split>& possibleSplits = GetBuffer<Split>(ePosSplits);
        nutty::Copy(possibleSplits.Begin(), _clearPosSplit->Begin(), possibleSplits.Size());

        nutty::Copy(m_prefixSumNodeCount.Begin(), m_null.Begin(), m_prefixSumNodeCount.Size());
        nutty::Copy(m_perThreadNodePos.Begin(), m_null.Begin(), m_perThreadNodePos.Size());
    }
};

std::vector<kdTree<float3>*> g_trees;

extern "C" void init(void)
{
    nutty::Init();
}

extern "C" void release(void)
{
    for(auto it = g_trees.begin(); it != g_trees.end(); ++it)
    {
        delete *it;
    }
    nutty::Release();
}

extern "C" IKDTree* generate(nutty::MappedPtr<float3>* data, uint elements, uint d, uint maxDepth)
{
    kdTree<float3>* tree = new kdTree<float3>(d, maxDepth);
    tree->Init(data, elements);
    tree->Generate();
    g_trees.push_back(tree);
    return tree;
}
