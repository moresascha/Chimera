#include "ParticleSystem.h"
#include <Wrap.h>
#include <Copy.h>
#include <Fill.h>
#include "util.h"

namespace chimera
{
    BaseEmitter::BaseEmitter(uint count, float starSpawnTime, float endSpawnTime)
        : m_particleCount(count), m_startSpawnTime(starSpawnTime), m_endSpawnTime(endSpawnTime), m_pParticleBuffer(NULL), m_pParticleVeloBuffer(NULL)
    {

    }

    MemPtr BaseEmitter::VGetParticleArray(void)
    {
        return (void*)m_mappedPositionPtr();
    }

    IVertexBuffer* BaseEmitter::VGetGFXPosArray(void)
    {
        return m_pParticleBuffer;
    }

    IVertexBuffer* BaseEmitter::VGetGFXVeloArray(void)
    {
        return m_pParticleVeloBuffer;
    }

    void BaseEmitter::VMapArrays(void)
    {
        /*m_mappedPositionPtr = m_mappedPositionResource.Bind();
        m_mappedVeloPtr = m_mappedVeloResource.Bind();*/
        size_t size;
        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsMapResources(1, &m_positionResoure));
        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsResourceGetMappedPointer((void**)m_mappedPositionPtr.GetRawPointerPtr(), &size, m_positionResoure));

        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsMapResources(1, &m_veloResoure));
        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsResourceGetMappedPointer((void**)m_mappedVeloPtr.GetRawPointerPtr(), &size, m_veloResoure));

        //DEBUG_OUT_A("posptr=%p veloptr=%p\n", m_positionResoure, m_veloResoure);
    }

    void BaseEmitter::VUnmapArrays(void)
    {
        /*m_mappedPositionResource.Unbind();
        m_mappedVeloResource.Unbind();*/
        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsUnmapResources(1, &m_positionResoure));
        CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsUnmapResources(1, &m_veloResoure));
    }

    MemPtr BaseEmitter::VGetAccelerationArray(void)
    {
        return m_acceleration.Begin()();
    }

    MemPtr BaseEmitter::VGetVelocitiesArray(void)
    {
        return m_mappedVeloPtr();
    }

    uint BaseEmitter::VGetByteCount(void) const
    {
        return (uint)
            (
            m_acceleration.Size()      * sizeof(float3) + 
            m_startingPositions.Size() * sizeof(float3) + 
            (size_t)m_pParticleBuffer->VGetByteCount()  +
            (size_t)m_pParticleVeloBuffer->VGetByteCount()
            );
    }

    void BaseEmitter::VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt)
    {
        float3 translation;
        ZeroMemory(&translation, sizeof(float3));
//         translation.x = sys->GetTranslation().x;
//         translation.y = sys->GetTranslation().y;
//         translation.z = sys->GetTranslation().z;

        void* ptr = VGetParticleArray();
        void* veloPtr = VGetVelocitiesArray();
        m_emitKernel.SetRawKernelArg(0, (void**)&ptr);
        m_emitKernel.SetRawKernelArg(2, (void**)&veloPtr);
        m_emitKernel.SetKernelArg(5, m_pos.m_v);
        m_emitKernel.SetKernelArg(6, time);
        m_emitKernel.SetKernelArg(7, dt);
        m_emitKernel.SetKernelArg(8, m_startSpawnTime);
        m_emitKernel.SetKernelArg(9, m_endSpawnTime);
        m_emitKernel.SetKernelArg(10, m_particleCount);
        m_emitKernel.Call();
    }

    void BaseEmitter::VCreateParticles(ParticlePosition* positions)
    {

    }

    void BaseEmitter::VRelease(void)
    {
        /*m_mappedPositionResource.Delete();
        m_mappedVeloResource.Delete();*/
        SAFE_DELETE(m_pParticleBuffer);
        SAFE_DELETE(m_pParticleVeloBuffer);
    }

    util::cmRNG rng;
    struct PosRand
    {
        ParticlePosition operator()()
        {
            ParticlePosition p;
            float scale = 1;
            p.alive = 0;
            p.x = rng.NextCubeFloat(scale);
            p.y = rng.NextFloat(scale);
            p.z = rng.NextCubeFloat(scale);
            return p;
        }
    };

    void BaseEmitter::VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter)
    {
        m_module.Create(KERNEL_PTX);

        m_emitKernel.Create(m_module.GetFunction("_computeEmitter"));

        PositiveNormalizedUniformValueGenerator rg(m_particleCount, 10);

        nutty::HostBuffer<float> rands(m_particleCount);
        rg.CreateRandomValues(rands.Begin()());
        m_emitterData.Resize(m_particleCount);

        for(uint i = 0; i < m_particleCount; ++i)
        {
            EmitterData d;
            d.rand = rands[i];
            d.time = 0;
            d.birthTime = 0;
            d.tmp = 0;
            m_emitterData.Insert(i, d);
        }

        nutty::HostBuffer<ParticlePosition> positions(m_particleCount);
        VCreateParticles(positions.Begin()());

        nutty::Fill(positions.Begin(), positions.End(), PosRand());

        m_startingPositions.Resize(m_particleCount);

        nutty::Copy(m_startingPositions.Begin(), positions.Begin(), m_particleCount);

        SAFE_DELETE(m_pParticleBuffer);

        m_pParticleBuffer = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer().release();

        m_pParticleBuffer->VInitParamater(m_particleCount, sizeof(ParticlePosition), positions.Begin()());

        m_pParticleBuffer->VCreate();

        //m_mappedPositionResource = nutty::WrapBuffer<ParticlePosition>((ID3D11Buffer*)m_pParticleBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone);
        assert(cudaGraphicsD3D11RegisterResource(&m_positionResoure, (ID3D11Buffer*)m_pParticleBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone) == cudaSuccess);

        float3 n = {};
        nutty::HostBuffer<float3> velos(m_particleCount);
        nutty::Fill(velos.Begin(), velos.End(), nutty::DefaultGenerator<float3>(n));

        SAFE_DELETE(m_pParticleVeloBuffer);

        m_pParticleVeloBuffer = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer().release();

        m_pParticleVeloBuffer->VInitParamater(m_particleCount, sizeof(float3), velos.Begin()());

        m_pParticleVeloBuffer->VCreate();

        //m_mappedVeloResource = nutty::WrapBuffer<float3>((ID3D11Buffer*)m_pParticleVeloBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone);
        assert(cudaGraphicsD3D11RegisterResource(&m_veloResoure, (ID3D11Buffer*)m_pParticleVeloBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone) == cudaSuccess);
//         cudaGraphicsResource_t res;
//         CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsD3D11RegisterResource(&res, (ID3D11Buffer*)m_pParticleBuffer->VGetDevicePtr(), cudaGraphicsMapFlagsNone));
//         CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsMapResources(1, &res));
//         void* devPtr;
//         size_t m_size;
//         CUDA_RT_SAFE_CALLING_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &m_size, res));


        uint grid = nutty::cuda::GetCudaGrid(m_particleCount, CUDA_GROUP_SIZE);
        m_emitKernel.SetDimension(grid, CUDA_GROUP_SIZE);

        m_acceleration.Resize(m_particleCount);
        nutty::Fill(m_acceleration.Begin(), m_acceleration.End(), n);

        m_emitKernel.SetKernelArg(1, m_startingPositions);
        m_emitKernel.SetKernelArg(3, m_acceleration);
        m_emitKernel.SetKernelArg(4, m_emitterData);
    }

    BaseEmitter::~BaseEmitter(void)
    {
        VRelease();
    }

//     PointEmitter::PointEmitter(const util::Vec3& position, uint particleCount, float start, float end) : BaseEmitter(position, particleCount, start, end)
//     {
// 
//     }
// 
//     ParticlePosition* PointEmitter::VCreateParticles(ParticlePosition* positions)
//     {
//         ParticlePosition* positions = new ParticlePosition[m_particleCount];
//         for(uint i = 0; i < m_particleCount; ++i)
//         {
//             positions[i].x = m_position.x;
//             positions[i].y = m_position.y;
//             positions[i].z = m_position.z;
//             positions[i].w = 0;
//         }
//         return positions;
//     }
// 
//     BoxEmitter::BoxEmitter(const util::Vec3& extends, const util::Vec3& position, uint particleCount, float start, float end) 
//         : BaseEmitter(position, particleCount, start, end), m_extends(extends)
//     {
// 
//     }
// 
//     ParticlePosition* BoxEmitter::CreateParticles(void)
//     {
//         ParticlePosition* positions = new ParticlePosition[m_particleCount];
//         srand(2);
//         for(uint i = 0; i < m_particleCount; ++i)
//         {
//             positions[i].x = m_position.x - m_extends.x + 2 * m_extends.x * rand() / (float)RAND_MAX;
//             positions[i].y = m_position.y - m_extends.y + 2 * m_extends.y * rand() / (float)RAND_MAX;;
//             positions[i].z = m_position.z - m_extends.z + 2 * m_extends.z * rand() / (float)RAND_MAX;;
//             positions[i].w = 0;
//         }
//         return positions;
//     }
// 
//     SurfaceEmitter::SurfaceEmitter(chimera::CMResource meshFile, const util::Vec3& position, uint particleCount, float start, float end) 
//         : BaseEmitter(position, particleCount, start, end), m_meshFile(meshFile)
//     {
// 
//     }
// 
//     util::Vec3 GetVertex(std::shared_ptr<chimera::Mesh> mesh, uint index)
//     {
//         uint stride = mesh->GetVertexStride() / 4;
//         float x = mesh->GetVertices()[index * stride + 0];
//         float y = mesh->GetVertices()[index * stride + 1];
//         float z = mesh->GetVertices()[index * stride + 2];
//         return util::Vec3(x, y, z);
//     }
// 
//     ParticlePosition* SurfaceEmitter::CreateParticles(void)
//     {
//         std::shared_ptr<chimera::Mesh> mesh = std::static_pointer_cast<chimera::Mesh>(chimera::g_pApp->GetCache()->GetHandle(m_meshFile));
//         const std::list<chimera::Face>& faces = mesh->GetFaces();
//         ParticlePosition* poses = new ParticlePosition[m_particleCount];
//         srand(10);
//         for(uint i = 0; i < m_particleCount; ++i)
//         {
//             util::Vec3 v0;
//             util::Vec3 v1;
//             util::Vec3 v2;
// 
//             int index = (int)((rand() / (float)RAND_MAX) * (mesh->GetFaces().size() - 1));
//             //DEBUG_OUT_A("%d, %d", index, mesh->GetFaces().size());
//             std::list<Face>::const_iterator it = faces.begin();
//             std::advance(it, index);
//             Face f = *it;
// 
//             if(f.m_triples.size() == 3)
//             {
//                 uint iv0 = f.m_triples[0].position;
//                 uint iv1 = f.m_triples[1].position;
//                 uint iv2 = f.m_triples[2].position;
// 
//                 v0 = GetVertex(mesh, iv0);
//                 v1 = GetVertex(mesh, iv1);
//                 v2 = GetVertex(mesh, iv2);
//             }
//             else if(f.m_triples.size() == 4)
//             {
//                 uint iv0 = f.m_triples[0].position;
//                 uint iv1 = f.m_triples[1].position;
//                 uint iv2 = f.m_triples[2].position;
//                 uint iv3 = f.m_triples[3].position;
//                 if(rand() / (float) RAND_MAX < 0.5)
//                 {
//                     v0 = GetVertex(mesh, iv0);
//                     v1 = GetVertex(mesh, iv1);
//                     v2 = GetVertex(mesh, iv2);
//                 }
//                 else
//                 {
//                     v0 = GetVertex(mesh, iv1);
//                     v1 = GetVertex(mesh, iv2);
//                     v2 = GetVertex(mesh, iv3);
//                 }
//             }
//             else
//             {
//                 LOG_CRITICAL_ERROR("SurfaceEmitter error, unknown triples size");
//             }
// 
//             float a = rand() / (float)RAND_MAX;
//             float b = (1-a) * rand() / (float)RAND_MAX;
//             float c = 1 - a - b;
// 
//             util::Vec3 pos = v0 * a + v1 * b + v2 * c;
//             ParticlePosition p(m_position.x + pos.x, m_position.y + pos.y, m_position.z + pos.z, 0);
//             poses[i] = p;
//         }
// 
//         return poses;
//     }
}