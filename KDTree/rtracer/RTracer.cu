#include "RTracer.h"
#include "../kernel.cuh"
#include "../../../Nutty/Nutty/Wrap.h"
#include "../../../Nutty/Nutty/Nutty.h"
#include "../../../Nutty/Nutty/DeviceBuffer.h"
#include "../../Source/chimera/Mat4.h"

#include <cutil_math.h>

#define PI 3.14159265
#define PI_MUL2 2*PI

texture<float4, cudaTextureType2D, cudaReadModeElementType> src;

struct Sphere
{
    float3 pos;
    float radius;
    float4 color;
};

__device__ float getAxis(float4 val, int axis) 
{
    switch(axis)
    {
        case 0 : return val.x;
        case 1 : return val.y;
        case 2 : return val.z;
        case 3 : return val.w;
    }
    return -1;
}

__device__ int intersectP(float4 eye, float4 ray, float4 invRay, float4 boxmin, float4 boxmax, float* tmin, float* tmax) {

    float t0 = 0; float t1 = FLT_MAX;

    for(uint i = 0; i < 3; ++i) 
    {
        float tNear = (getAxis(boxmin, i) - getAxis(eye, i)) * getAxis(invRay, i);
        float tFar = (getAxis(boxmax, i) - getAxis(eye, i)) * getAxis(invRay, i);

        if(tNear > tFar) 
        {
            float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;

        if(t0 > t1) return 0;
    }
    *tmin = t0;
    *tmax = t1;
    return 1;
}

//seite 118 in PBRT, Compute quadratic sphere coeffficienten
__device__ float4 computeQuadraticCoefs(float4 eye, float4 ray, float radius) 
{
    float4 v;
    v.x = 1;//ray.x*ray.x + ray.y*ray.y + ray.z*ray.z;
    v.y = 2 * (ray.x*eye.x + ray.y*eye.y + ray.z*eye.z);
    v.z = eye.x*eye.x + eye.y*eye.y + eye.z*eye.z - radius*radius;
    return v;
}

//seite 118 in PVRT, Quadtratic
__device__ int quad(float A, float B, float C, float* t0, float* t1) 
{
    float d = B * B - 4.0 * A * C;
    if(d <= 0) return 0;
    float rd = sqrt(d);
    float q;
    if(B < 0) 
    {
        q = -0.5 * (B - rd);
    } else {
        q = -0.5 * (B + rd);
    }
    float x = q / A;
    float y = C / q;
    if(x > y) 
    {
        *t0 = y;
        *t1 = x;
    } else {
        *t0 = x;
        *t1 = y;
    }
    return 1;
}

//119 in PBRT, Compute intersection distance along ray
__device__ int isHit(float t0, float t1, float n, float f, float* hit) 
{
    if(t0 > f || t1 < n) return 0;
    *hit = t0;
    if(t0 < n) 
    {
        *hit = t1;
        if(*hit > f) return 0;
    }
    return 1;
}

__device__ void mul(float* m4x4l, float* m4x4r, float* result)
{
    float4 r0 = make_float4(*m4x4l, *(m4x4l+1), *(m4x4l+2), *(m4x4l+3));
    float4 r1 = make_float4(*(m4x4l+4), *(m4x4l+5), *(m4x4l+6), *(m4x4l+7));
    float4 r2 = make_float4(*(m4x4l+8), *(m4x4l+9), *(m4x4l+10), *(m4x4l+11));
    float4 r3 = make_float4(*(m4x4l+12), *(m4x4l+13), *(m4x4l+14), *(m4x4l+15));

    float4 c0 = make_float4(*m4x4r, *(m4x4r+1), *(m4x4r+2), *(m4x4r+3));
    float4 c1 = make_float4(*(m4x4r+4), *(m4x4r+5), *(m4x4r+6), *(m4x4r+7));
    float4 c2 = make_float4(*(m4x4r+8), *(m4x4l+9), *(m4x4r+10), *(m4x4r+11));
    float4 c3 = make_float4(*(m4x4r+12), *(m4x4r+13), *(m4x4r+14), *(m4x4r+15));

    *(result+0) = dot(r0, c0);
    *(result+1) = dot(r0, c1);
    *(result+2) = dot(r0, c2);
    *(result+3) = dot(r0, c3);

    *(result+4) = dot(r1, c0);
    *(result+5) = dot(r1, c1);
    *(result+6) = dot(r1, c2);
    *(result+7) = dot(r1, c3);

    *(result+8) = dot(r2, c0);
    *(result+9) = dot(r2, c1);
    *(result+10) = dot(r2, c2);
    *(result+11) = dot(r2, c3);

    *(result+12) = dot(r3, c0);
    *(result+13) = dot(r3, c1);
    *(result+14) = dot(r3, c2);
    *(result+15) = dot(r3, c3);
}

__device__ float4 transform(float* m4x4l, float4 vector)
{
    float4 r0 = make_float4(*m4x4l, *(m4x4l+4), *(m4x4l+8), *(m4x4l+12));
    float4 r1 = make_float4(*(m4x4l+1), *(m4x4l+5), *(m4x4l+9), *(m4x4l+13));
    float4 r2 = make_float4(*(m4x4l+2), *(m4x4l+6), *(m4x4l+10), *(m4x4l+14));
    float4 r3 = make_float4(*(m4x4l+3), *(m4x4l+7), *(m4x4l+11), *(m4x4l+15));

    return make_float4(dot(r0, vector), dot(r1, vector), dot(r2, vector), dot(r3, vector));
}

__device__ float4 refract(float4 i, float4 n, float eta)
{
  float cosi = dot(-i, n);
  float cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
  float4 t = eta*i + ((eta*cosi - sqrt(abs(cost2))) * n);
  return t * make_float4(cost2 > 0);
}

struct HitResult
{
    float4 normal;
    float4 worldPosition;
    float t;
    int isHit;
};

__device__ int computeHit(float3 sphere, float4 eye, float4 ray, float tmin, float tmax, float radius, HitResult* result)
{
    float4 sphereToEye = eye - make_float4(sphere.x, sphere.y, sphere.z, 0);
    float4 cos = computeQuadraticCoefs(sphereToEye, ray, radius);
    float t0, t1;
    if(!quad(cos.x, cos.y, cos.z, &t0, &t1)) 
    {
        return 0;
    }

    float t;
    if(!isHit(t0, t1, tmin, tmax, &t))
    {
        return 0;
    }

    if(t > result->t)
    {
        return 0;
    }

    result->worldPosition = (eye + ray * t);
    result->normal = normalize(result->worldPosition - make_float4(sphere.x, sphere.y, sphere.z, 0));
    result->t = t;
    result->isHit = 1;

    return 1;
}

struct ToDo 
{
    Split node;
    float min;
    float max;
    uint treePos;
    uint level;
};

__device__ void traversal(float3* spheres, Split* splits, uint* contentCount, float4 eye, float4 ray, HitResult* hit, float4 min, float4 max, uint treeDepth) 
{
    float tmin, tmax;

    hit->isHit = 0;

    if(!intersectP(eye, ray, 1.0/ray, min, max, &tmin, &tmax))
    {
        return;
    }

    if(tmin < 0.0f) tmax = 0.0f;

    hit->t = FLT_MAX;

    ToDo toDo[16];

    Split n = splits[0];

    int pos = 0; 

    float4 invDir = make_float4(1.0 / ray.x, 1.0 / ray.y, 1.0 / ray.z, 0);

    uint treePos = 0;
    uint level = 0;

    while(level < treeDepth) 
    {
        if(hit->t < tmin) 
        {
            break;
        }

        if(contentCount[treePos] > 2 && level < (treeDepth-1)) 
        {
            int axis = n.axis;

            float tplane = (n.split - getAxis(eye, axis)) * getAxis(invDir, axis);

            Split c0; Split c1;

            int belowFirst = (getAxis(eye, axis) < n.split) || ((getAxis(eye, axis) == n.split) && (getAxis(ray, axis) >= 0));

            if(belowFirst)
            {
                c0 = splits[treePos + (1 << level)];
                c1 = splits[treePos + (1 << level)+1];
            } else 
            {
                c0 = splits[treePos + (1 << level)+1];
                c1 = splits[treePos + (1 << level)];
            }
            if(tplane > tmax || tplane <= 0)
            {
                n = c0;
                treePos += (1 << level);
            } else if(tplane < tmin) 
            {
                n = c1;
                treePos += (1 << level)+1;
            } else 
            {
//                 ToDo td;
//                 td.treePos = treePos + (1 << level)+1;
//                 td.level = level;
//                 td.node = c1;
//                 td.min = tplane;
//                 td.max = tmax;
//                 toDo[pos] = td;
//                 pos++;
//                 n = c0;
//                 treePos += (1 << level);
//                 tmax = tplane;
            }
            level++;
        } else
        {
            uint spheresCnt = contentCount[treePos];
            for(int i = 0; i < spheresCnt; ++i)
            {
                computeHit(spheres[n.contentStartIndex + i], eye, ray, tmin, tmax, 1, hit);
            }
            if(pos > 0) 
            {
//                 pos -= 1;
//                 n = toDo[pos].node;
//                 tmin = toDo[pos].min;
//                 tmax = toDo[pos].max;
//                 treePos = toDo[pos].treePos;
//                 level = toDo[pos].level;
            } else 
            {
                break;
            }
        }
    }
}

__global__ void simpleSphereTracer(
    float4* dst, 
    float3* spheres,
    float3* aabbs,
    uint* nodesContentCount,
    Split* splits,
    uint treeDepth,
    float* view, float3 _eye, uint w, uint h)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    uint N = w * h;

    if(id >= N)
    {
        return;
    }
    
    uint tilexos = idx / (blockDim.x * blockDim.y);
    uint tileyos = blockIdx.y * (blockDim.x * gridDim.x * blockDim.y);

    //id = (blockDim.x * gridDim.x * blockDim.y) * (tileyos + idx / blockDim.x) + (tilexos + threadIdx.x);
    
    float u = idx / (float)w;//(tilexos + threadIdx.x) / (float)w;
    float v = idy / (float)h;//(tileyos + idx / blockDim.x) / (float)h;

    float4 eye = make_float4(_eye, 0);

    float4 color = tex2D(src, u, v);

    float4 ray = normalize(make_float4(2 * u - 1, 2 * (1-v) - 1, 1.0f, 0));

    ray = transform(view, ray);
    ray.w = 0;

    float4 mini = make_float4(aabbs[0], 0);
    float4 maxi = make_float4(aabbs[1], 0);
    mini -= make_float4(1,1,1,0);
    maxi += make_float4(1,1,1,0);
    
    HitResult res;
    memset(&res, 0, sizeof(HitResult));
    res.t = FLT_MAX;

    traversal(spheres, splits, nodesContentCount, eye, ray, &res, mini, maxi, treeDepth);

    if(!res.isHit)
    {
        dst[id] = color;
        return;
    }

    float4 light = make_float4(1.0f,0.3f,-0.2f,0);

    float4 rfract = normalize(refract(normalize(ray * res.t), res.normal, 1.0/1.5));

    rfract = rfract * 0.5 + 0.5;
    rfract.y = 1 - rfract.y;

    float4 refractionColor = tex2D(src, rfract.x, rfract.y);
   
    float4 c = 1.05*refractionColor;// * lerp(make_float4(0.1,0.1,0.1, 0), make_float4(1,1,1,1), dot(light, normal));

    dst[id] = make_float4(c.x, c.y, c.z, color.w);
}

class wtf_tracer : public IRTracer
{
private:
    //nutty::MappedTexturePtr<float4> m_dst;
    cudaGraphicsResource_t m_dst;
    nutty::DeviceBuffer<float3>* m_kdData;
    nutty::DeviceBuffer<float3>* m_kdBBox;
    nutty::DeviceBuffer<Split>* m_splits;
    nutty::DeviceBuffer<uint>* m_contentCount;
    uint m_width;
    uint m_height;
    void* m_linearMem;
    size_t m_pitch;

    nutty::DeviceBuffer<float> m_view;
    nutty::DeviceBuffer<float3> m_eye;

    IKDTree* m_tree;

public:
    wtf_tracer(IKDTree* tree);

    void VRender(void);

    BOOL VOnRestore(UINT w, UINT h);

    ~wtf_tracer(void);
};

wtf_tracer::wtf_tracer(IKDTree* tree) 
    : IRTracer("wtf_tracer"), m_width(800), m_height(600), m_linearMem(NULL), m_tree(tree)
{
    m_view.Resize(16);
    m_eye.Resize(1);

    m_kdData = (nutty::DeviceBuffer<float3>*)m_tree->GetData();
    m_kdBBox = (nutty::DeviceBuffer<float3>*)m_tree->GetBuffer(eAxisAlignedBB);
    m_splits = (nutty::DeviceBuffer<Split>*)m_tree->GetBuffer(eSplitData);
    m_contentCount = (nutty::DeviceBuffer<uint>*)m_tree->GetBuffer(eNodesContentCount);

}

IRTracer* createTracer(IKDTree* tree, int flags)
{
    return new wtf_tracer(tree);
}

void wtf_tracer::VRender(void)
{
    //nutty::DevicePtr<float4> ptr = m_dst.Bind();
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &m_dst));
    cudaArray_t ptr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&ptr, m_dst, 0, 0));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    float* view = (float*)&chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VGetCamera()->GetIView().m_m;
    float* eye = (float*)&chimera::CmGetApp()->VGetHumanView()->VGetSceneGraph()->VGetCamera()->GetEyePos().m_v;

    CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpy(m_view.Begin()(), view, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpy(m_eye.Begin()(), eye, sizeof(float3), cudaMemcpyHostToDevice));

    size_t os = 0;

    src.addressMode[0] = cudaAddressModeWrap;
    src.addressMode[1] = cudaAddressModeWrap;
    src.normalized = 1;
    src.filterMode = cudaFilterModeLinear;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(&src, ptr, &desc));
    dim3 g;
    g.x = nutty::cuda::getCudaGrid(m_width, 16U);
    g.y = nutty::cuda::getCudaGrid(m_height, 16U);
    dim3 tiles;
    tiles.x = 16;
    tiles.y = 16;

    simpleSphereTracer<<<g, tiles>>>(
        (float4*)m_linearMem,
        m_kdData->Begin()(),
        m_kdBBox->Begin()(), 
        m_contentCount->Begin()(),
        m_splits->Begin()(),
        
        m_tree->GetCurrentDepth(), m_view.Begin()(), *m_eye.Begin(), m_width, m_height);

    CUDA_RT_SAFE_CALLING_SYNC(cudaUnbindTexture(&src));

    CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpy2DToArray(ptr, 0, 0, m_linearMem, m_pitch, m_width * sizeof(float4), m_height, cudaMemcpyDeviceToDevice));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &m_dst));
    //m_dst.Unbind();
}

BOOL wtf_tracer::VOnRestore(UINT w, UINT h)
{
    m_width = w;
    m_height = h;

    if(m_linearMem)
    {
        cudaFree(m_linearMem);
    }
    
    CUDA_RT_SAFE_CALLING_SYNC(cudaMallocPitch(&m_linearMem, &m_pitch, m_width* sizeof(float4), m_height));
    
    CUDA_RT_SAFE_CALLING_SYNC(cudaMemset(m_linearMem, 0, m_pitch * m_height));

    chimera::IDeviceTexture* colorBuffer = chimera::CmGetApp()->VGetRenderer()->VGetCurrentRenderTarget()->VGetTexture();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsD3D11RegisterResource(&m_dst, (ID3D11Texture2D*)colorBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone));

    //m_dst = std::move(nutty::WrapTexture<float4>((ID3D11Texture2D*)colorBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone));

    return TRUE;
}

wtf_tracer::~wtf_tracer(void)
{
    if(m_linearMem)
    {
        cudaFree(m_linearMem);
    }
}