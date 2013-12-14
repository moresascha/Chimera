#include "tracer.cuh"
#include "../kdtree.cuh"
#include "../../../Nutty/Nutty/Wrap.h"
#include "../../../Nutty/Nutty/cuda/Module.h"
#include "../../../Nutty/Nutty/cuda/Kernel.h"
#include "../../../Nutty/Nutty/Nutty.h"
#include "../../../Nutty/Nutty/Copy.h"
#include "../../../Nutty/Nutty/Fill.h"
#include "../../../Nutty/Nutty/Scan.h"
#include "../../../Nutty/Nutty/cuda/Stream.h"
#include "../../../Nutty/Nutty/DeviceBuffer.h"
#include "../../../Nutty/Nutty/cuda/cuda_helper.h"
#include "../../Source/chimera/Mat4.h"
#include "../../Source/chimera/Event.h"
#include "../Source/chimera/Timer.h"
#include "../DoubleBuffer.h"

#include <cutil_math.h>

class wtf_tracer : public IRTracer
{
private:
    //nutty::MappedTexturePtr<float4> m_dst;
    cudaGraphicsResource_t m_dst;
    cudaGraphicsResource_t m_worldPosition;
    nutty::DeviceBuffer<float3>* m_kdData;
    nutty::DeviceBuffer<AABB>* m_kdBBox;
    nutty::DeviceBuffer<Node>* m_nodes;

    nutty::DeviceBuffer<Ray> m_initialRays;
    nutty::DeviceBuffer<uint> m_initRayMask;
    nutty::DeviceBuffer<uint> m_scannedRayMask;
    nutty::DeviceBuffer<uint> m_sums;

    nutty::DeviceBuffer<Ray> m_shadowRays;
    nutty::DeviceBuffer<uint> m_shadowRayMask;

    uint m_width;
    uint m_height;
    void* m_linearMem;
    size_t m_pitch;

    nutty::cuModule m_module;
    nutty::cuKernel m_kernel;
    nutty::cuKernel m_computeInitialrays;
    nutty::cuKernel m_computeRays;

    nutty::cuTexRef m_frameBufferRef;
    nutty::cuTexRef m_worldPositionsRef;

    nutty::DeviceBuffer<float> m_view;

    IKDTree* m_tree;
    int m_enable;
    uint m_lastRayCount;
    chimera::util::HTimer m_timer;

public:
    wtf_tracer(IKDTree* tree);

    void VRender(void);

    int VOnRestore(uint w, uint h);

    void ReleaseSharedResources(void);

    uint GetLastRayCount(void) { return m_lastRayCount; }

    uint GetLastShadowRayCount(void) { return 0; }

    void ToggleEnable(void);

    void Compile(void);

    double GetLastMillis(void);

    ~wtf_tracer(void);
};

wtf_tracer::wtf_tracer(IKDTree* tree) 
    : IRTracer("wtf_tracer"), m_width(800), m_height(600), m_linearMem(NULL), m_tree(tree), m_enable(TRUE), m_worldPosition(NULL), m_dst(NULL), m_lastRayCount(0)
{
    m_view.Resize(16);
    m_kdBBox = m_tree->GetAABBs();
    m_nodes = m_tree->GetNodes();
    m_kdData = (nutty::DeviceBuffer<float3>*)m_tree->GetData();
}

double wtf_tracer::GetLastMillis(void)
{
    return m_timer.GetMillis();
}

void wtf_tracer::ToggleEnable(void)
{
    m_enable = !m_enable;
}

void wtf_tracer::ReleaseSharedResources(void)
{
    if(m_dst)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnregisterResource(m_dst));
        m_dst = NULL;
    }

    if(m_worldPosition)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnregisterResource(m_worldPosition));
        m_worldPosition = NULL;
    }
}

void wtf_tracer::VRender(void)
{
    if(!m_enable)
    {
        return;
    }
    
    chimera::CmGetApp()->VGetRenderer()->VPresent();

    cudaDeviceSynchronize();
    m_timer.Start();
    
    CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsMapResources(1, &m_dst, m_tree->GetDefaultStream()()));
    cudaArray_t ptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&ptr, m_dst, 0, 0));

    CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsMapResources(1, &m_worldPosition, m_tree->GetDefaultStream()()));
    cudaArray_t worldPosptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&worldPosptr, m_worldPosition, 0, 0));

    float* view = (float*)&chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VGetCamera()->GetIView().m_m;
    XMFLOAT3 eye = chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VGetCamera()->GetEyePos().m_v;

    CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpy(m_view.Begin()(), view, 16 * sizeof(float), cudaMemcpyHostToDevice));

    m_frameBufferRef.BindToArray((CUarray)ptr);
    m_worldPositionsRef.BindToArray((CUarray)worldPosptr);

    dim3 g;
    g.x = nutty::cuda::GetCudaGrid(m_width, 16U);
    g.y = nutty::cuda::GetCudaGrid(m_height, 16U);
    g.z = 1;
    dim3 tiles;
    tiles.x = 16;
    tiles.y = 16;
    tiles.z = 1;

    uint depth = m_tree->GetCurrentDepth();
    /*m_kernel.SetKernelArg(4, depth);
    m_kernel.SetKernelArg(6, eye);
    m_kernel.SetKernelArg(7, m_width);
    m_kernel.SetKernelArg(8, m_height);
    m_kernel.SetDimension(g, tiles);*/

    nutty::ZeroMem(m_initRayMask);
    nutty::ZeroMem(m_scannedRayMask);
    nutty::ZeroMem(m_sums);

    m_computeInitialrays.SetKernelArg(3, eye);
    m_computeInitialrays.SetKernelArg(8, m_width);
    m_computeInitialrays.SetKernelArg(9, m_height);
    m_computeInitialrays.SetDimension(g, tiles);
    m_computeInitialrays.Call(m_tree->GetDefaultStream()());

    for(int i = 0; i < 3; ++i)
    {       
        CUDA_SAFE_THREAD_SYNC();
        nutty::SetStream(m_tree->GetDefaultStream());
        nutty::PrefixSumScan(m_initRayMask.Begin(), m_initRayMask.End(), m_scannedRayMask.Begin(), m_sums.Begin());
        CUDA_SAFE_THREAD_SYNC();

        nutty::Compact(m_initialRays.Begin(), m_initialRays.End(), m_initRayMask.Begin(), m_scannedRayMask.Begin(), 0U);
        CUDA_SAFE_THREAD_SYNC();

        m_lastRayCount = *(m_scannedRayMask.End()-1);
    
        //DEBUG_OUT_A("%d %d\n", m_lastRayCount, sum);
    
        if(m_lastRayCount > 0)
        {
            uint blockSize = 256;
            g = nutty::cuda::GetCudaGrid(m_lastRayCount, blockSize);
            m_computeRays.SetDimension(g, blockSize);
            m_computeRays.SetKernelArg(5, depth);
            m_computeRays.SetKernelArg(6, m_width);
            m_computeRays.SetKernelArg(7, m_lastRayCount);
            m_computeRays.Call(m_tree->GetDefaultStream()());
        }
        else
        {
            break;
        }
    }

    cudaDeviceSynchronize();

    /*nutty::HostBuffer<uint> cpy(m_width * m_height);

    nutty::Copy(cpy.Begin(), m_initRayMask.Begin(), m_width * m_height);
    uint sum = 0;
    for(int i = 0; i < cpy.Size(); ++i)
    {
        uint t = cpy[i];
        sum += t;
    }*/

    CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpy2DToArray(ptr, 0, 0, m_linearMem, m_pitch, m_width * sizeof(float4), m_height, cudaMemcpyDeviceToDevice));
    CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsUnmapResources(1, &m_dst, m_tree->GetDefaultStream()()));
    CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsUnmapResources(1, &m_worldPosition, m_tree->GetDefaultStream()()));

    m_timer.Stop();
}

BOOL wtf_tracer::VOnRestore(UINT w, UINT h)
{
    m_width = w;
    m_height = h;

    dim3 dim;
    dim.x = nutty::cuda::GetCudaGrid(m_width, 16U);
    dim.y = nutty::cuda::GetCudaGrid(m_height, 16U);

    m_initialRays.Resize(dim.x * dim.y * 16 * 16);
    m_initRayMask.Resize(dim.x * dim.y * 16 * 16);
    m_scannedRayMask.Resize(dim.x * dim.y * 16 * 16);
    m_sums.Resize((m_scannedRayMask.Size()) / 512);

    m_shadowRayMask.Resize(w * h);
    m_shadowRays.Resize(w * h);

    if(m_linearMem)
    {
        cudaFree(m_linearMem);
    }
    
    CUDA_RT_SAFE_CALLING_SYNC(cudaMallocPitch(&m_linearMem, &m_pitch, m_width* sizeof(float4), m_height));
    
    CUDA_RT_SAFE_CALLING_SYNC(cudaMemset(m_linearMem, 0, m_pitch * m_height));

    chimera::IDeviceTexture* colorBuffer = chimera::CmGetApp()->VGetRenderer()->VGetCurrentRenderTarget()->VGetTexture();
    chimera::IDeviceTexture* worldPosition = chimera::CmGetApp()->VGetRenderer()->VGetAlbedoBuffer()->VGetRenderTarget(chimera::eDiff_WorldPositionTarget)->VGetTexture();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsD3D11RegisterResource(&m_dst, (ID3D11Texture2D*)colorBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsD3D11RegisterResource(&m_worldPosition, (ID3D11Texture2D*)worldPosition->VGetDevicePtr(), cudaGraphicsMapFlagsNone));

    Compile();

    return TRUE;
}

void wtf_tracer::Compile(void)
{
    /*nutty::cuModule test;
    test.Create("ptx/tracer_kernel.ptx");*/

    m_module.Create("ptx/tracer_kernel.ptx");
    m_kernel.Create(m_module.GetFunction("simpleSphereTracer"));

    m_kernel.SetKernelArg(0, m_linearMem);
    m_kernel.SetKernelArg(1, *m_kdData);
    m_kernel.SetKernelArg(2, *m_kdBBox);
    m_kernel.SetKernelArg(3, *m_nodes);
    m_kernel.SetKernelArg(5, m_view);

    m_computeInitialrays.Create(m_module.GetFunction("computeInitialRays"));
    m_computeInitialrays.SetKernelArg(0, m_linearMem);
    m_computeInitialrays.SetKernelArg(1, *m_kdBBox);
    m_computeInitialrays.SetKernelArg(2, m_view);
    m_computeInitialrays.SetKernelArg(4, m_initialRays);
    m_computeInitialrays.SetKernelArg(5, m_shadowRays);
    m_computeInitialrays.SetKernelArg(6, m_initRayMask);
    m_computeInitialrays.SetKernelArg(7, m_shadowRayMask);

    m_computeRays.Create(m_module.GetFunction("computeRays"));
    m_computeRays.SetKernelArg(0, m_linearMem);
    m_computeRays.SetKernelArg(1, m_initialRays);
    m_computeRays.SetKernelArg(2, m_initRayMask);
    m_computeRays.SetKernelArg(3, *m_nodes);
    m_computeRays.SetKernelArg(4, *m_kdData);

    m_frameBufferRef = m_module.GetTexRef("src");
    m_frameBufferRef.NormalizedCoords();
    m_frameBufferRef.SetFilterMode(CU_TR_FILTER_MODE_LINEAR);
    m_frameBufferRef.SetFormat(CU_AD_FORMAT_FLOAT, 4);
    m_frameBufferRef.SetAddressMode(CU_TR_ADDRESS_MODE_WRAP, 0);
    m_frameBufferRef.SetAddressMode(CU_TR_ADDRESS_MODE_WRAP, 1);

    m_worldPositionsRef = m_module.GetTexRef("worldPosTexture");
    m_worldPositionsRef.NormalizedCoords();
    m_worldPositionsRef.SetFilterMode(CU_TR_FILTER_MODE_LINEAR);
    m_worldPositionsRef.SetFormat(CU_AD_FORMAT_FLOAT, 4);
    m_worldPositionsRef.SetAddressMode(CU_TR_ADDRESS_MODE_WRAP, 0);
    m_worldPositionsRef.SetAddressMode(CU_TR_ADDRESS_MODE_WRAP, 1);
}

wtf_tracer::~wtf_tracer(void)
{
    if(m_linearMem)
    {
        cudaFree(m_linearMem);
    }
}

IRTracer* createTracer(IKDTree* tree, int flags)
{
    return new wtf_tracer(tree);
}
