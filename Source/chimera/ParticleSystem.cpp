#include "ParticleSystem.h"
#include <algorithm>
#include "GameApp.h"
#include "VRamManager.h"
#include "GameView.h"
#include "Geometry.h"
#include "math.h"

namespace chimera
{
    VRamHandle* ParticleQuadGeometryHandleCreator::VGetHandle(void)
    {
        return new chimera::Geometry(false);
    }

    void ParticleQuadGeometryHandleCreator::VCreateHandle(VRamHandle* handle)
    {

        chimera::Geometry* geo = (chimera::Geometry*)handle;

        float qscale = 0.01f;

        float quad[32] = 
        {
            -1 * qscale, -1 * qscale, 0, 0,1,0, 0, 0, 
            +1 * qscale, -1 * qscale, 0, 0,1,0, 1, 0,
            -1 * qscale, +1 * qscale, 0, 0,1,0, 0, 1,
            +1 * qscale, +1 * qscale, 0, 0,1,0, 1, 1
        };

        uint indices[4] = 
        {
            0, 1, 2, 3
        };

        geo->SetVertexBuffer(quad, 4, 32);
        geo->SetIndexBuffer(indices, 4, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->VCreate();
    }

    IGeometry* ParticleSystem::VGetGeometry(void)
    {
        if(!m_geometry->VIsReady())
        {
            m_geometry = std::static_pointer_cast<Geometry>(CmGetApp()->VGetVRamManager()->VGetHandle(VRamResource(".ParticleQuadGeometry")));
        }
        return m_geometry.get();
    }

    ParticleSystem::ParticleSystem(RNG generator) :
        m_fpRandGenerator(generator), m_time(0), m_updateInterval(16), m_localWorkSize(256), m_updateInterval(16)
    {
    }

    void ParticleSystem::AddModifier(BaseModifier* mod)
    {
        m_mods.push_back(mod);
        mod->SetAABB(m_aabb);
    }

    void ParticleSystem::RemoveModifier(BaseModifier* mod)
    {
        auto it = std::find(m_mods.begin(), m_mods.end(), mod);
        if(it != m_mods.end())
        {
            BaseModifier* mod = *it;
            m_mods.erase(it);
            SAFE_DELETE(mod);
        }
    }

    const util::Vec3& ParticleSystem::GetTranslation(void)
    {
        return m_position;
    }

    void ParticleSystem::VUpdateTick(ulong time, ulong dt)
    {
        //m_pCuda->MapGraphicsResource(m_particles);
        nutty::MappedPtr<float4> ptr = m_particles->Bind();

        //assert((INT)timer->GetTime() - (INT)m_startTime > 0);
        dt = CLAMP(dt, dt, m_updateInterval);
        m_time += dt;
        float dtt = (float)(dt * 1e-3f);
        float tf = m_time * 1e-3f;
        //m_pEmitter->VUpdate(this, tf, dtt);

        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            IParticleModifier* mod = *it;
            mod->VUpdate(this, tf, dtt);
        }

        void *args[] = { &m_particles->ptr, &m_acceleration->ptr, &m_velocities->ptr, &dtt };

        m_kernel->m_ppArgs = args;

        m_kernel->Call();
        //m_pCuda->CallKernel(m_kernel);
        //integrate((float4*)m_particles->ptr, (float3*)m_acceleration->ptr, (float3*)m_velocities->ptr, (float)(dt * 1e-3f), GetParticlesCount(), GetLocalWorkSize());

        //DEBUG_OUT_A("Count=%d Lws=%u Blocks=%u", GetParticlesCount(), GetLocalWorkSize(), GetParticlesCount() / GetLocalWorkSize());

        //m_pCuda->UnmapGraphicsResource(m_particles);

        m_particles->Unbind();
    }

    LPCGPUDEVMEM ParticleSystem::VGetDeviceRNGArray(void)
    {
        return &m_randomValues;
    }

    LPCGPUDEVMEM ParticleSystem::VGetDeviceAccelerationArray(void)
    {
        return &m_acceleration;
    }

    LPCGPUDEVMEM ParticleSystem::VGetDeviceVelocitiesArray(void)
    {
        return &m_velocities;
    }

    cudah::cuda_buffer ParticleSystem::GetParticles(void)
    {
        return m_particles;
    }

    uint ParticleSystem::GetParticlesCount(void)
    {
        return m_pEmitter->GetParticleCount();
    }

    bool ParticleSystem::HasRandBuffer(void)
    {
        return m_randomValues != NULL;
    }

    void ParticleSystem::SetAxisAlignedBB(util::AxisAlignedBB& aabb)
    {
        m_aabb = aabb;
        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            (*it)->SetAABB(m_aabb);
        }
        if(m_pEmitter)
        {
            m_pEmitter->SetAABB(aabb);
        }
    }

    util::AxisAlignedBB& ParticleSystem::GetAxisAlignedBB(void)
    {
        return m_aabb;
    }

    chimera::VertexBuffer* ParticleSystem::GetParticleBuffer(void)
    {
        return m_pParticleBuffer;
    }

    uint ParticleSystem::GetLocalWorkSize(void)
    {
        return m_localWorkSize;
    }

    //vram interface
    bool ParticleSystem::VCreate(void)
    {
        SAFE_DELETE(m_pCuda);

        m_pCuda = new cudah::cudah("./chimera/ptx/Particles.ptx");

        m_kernel = m_pCuda->GetKernel("_integrate");

        int blockSize = 256;//todo cudah::cudah::GetMaxThreadsPerSM() / cudah::cudah::GetMaxBlocksPerSM();
        m_localWorkSize = blockSize;

        if(m_pEmitter->GetParticleCount() % m_localWorkSize != 0)
        {
            LOG_CRITICAL_ERROR("particle count not supported");
        }
        
        ParticlePosition* positions = m_pEmitter->CreateParticles();
        float* rands = NULL;

        if(m_pRandGenerator)
        {
            rands = m_pRandGenerator->CreateRandomValues();
            m_randomValues = m_pCuda->CreateBuffer(std::string("randoms"), m_pRandGenerator->GetValuesCount() * 4, rands, 4);
        }

        float* parts = (float*)positions;

        SAFE_DELETE(m_pParticleBuffer);

        m_pParticleBuffer = new chimera::VertexBuffer(parts, m_pEmitter->GetParticleCount(), 16);

        m_pParticleBuffer->Create();

        m_particles = m_pCuda->RegisterD3D11Buffer(std::string("emitterGeometry"), m_pParticleBuffer->GetBuffer(), cudaGraphicsMapFlagsNone);

        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            BaseModifier* mod = *it;
            mod->VOnRestore(this);
        }

        m_pEmitter->VOnRestore(this);

        m_pEmitter->SetAABB(m_aabb);

        float3* veloAccInit = new float3[m_pEmitter->GetParticleCount()];
        for(uint i = 0; i < m_pEmitter->GetParticleCount(); ++i)
        {
            veloAccInit[i].x = 0;
            veloAccInit[i].y = 0;
            veloAccInit[i].z = 0;
        }

        m_acceleration = m_pCuda->CreateBuffer(std::string("acc"), m_pEmitter->GetParticleCount() * 3 * 4, veloAccInit, 12);

        m_velocities = m_pCuda->CreateBuffer(std::string("velo"), m_pEmitter->GetParticleCount() * 3 * 4, veloAccInit, 12);

        m_geometry = std::static_pointer_cast<chimera::Geometry>(chimera::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(chimera::VRamResource(".ParticleQuadGeometry")));

        //integrate((float4*)m_particles->ptr, (float3*)m_acceleration->ptr, (float3*)m_velocities->ptr, (float)(dt * 1e-3f), GetParticlesCount(), GetLocalWorkSize());
        int threads = cudahu::GetThreadCount(GetParticlesCount(), GetLocalWorkSize());
        m_kernel->m_blockDim.x = GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        SAFE_ARRAY_DELETE(veloAccInit);
        SAFE_ARRAY_DELETE(rands);
        SAFE_ARRAY_DELETE(positions);

        m_time = 0;

        return true;
    }

    void ParticleSystem::VDestroy(void)
    {
        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            BaseModifier* mod = *it;
            SAFE_DELETE(mod);
        }
        SAFE_DELETE(m_pParticleBuffer);
        SAFE_DELETE(m_pCuda);
        SAFE_DELETE(m_pEmitter);
        SAFE_DELETE(m_pRandGenerator);
    }

    uint ParticleSystem::VGetByteCount(void) const
    {
        //TODO modifier
        //per particle
        //acceleration 3 * 4 byte
        //velocity 3 * 4 byte
        //position 4 * 4 byte
        uint modBytes = 0;
        TBD_FOR(m_mods)
        {
            modBytes += (*it)->VGetByteCount();
        }
        uint particles = m_pEmitter->GetParticleCount();
        uint byte = sizeof(float);
        return particles * (3 + 3 + 4) * byte + modBytes;
    }

    ParticleSystem::~ParticleSystem(void)
    {

    }

    PositiveNormalizedUniformValueGenerator::PositiveNormalizedUniformValueGenerator(uint count, int seed) : IRandomGenerator(count), m_seed(seed)
    {

    }

    float* PositiveNormalizedUniformValueGenerator::CreateRandomValues(void)
    {
        srand(m_seed);
        float* rands = new float[m_count];
        for(uint i = 0; i < m_count; ++i)
        {
            rands[i] = rand() / (float)RAND_MAX;
        }
        return rands;
    }

    NormalizedUniformValueGenerator::NormalizedUniformValueGenerator(uint count, int seed, float scale) : IRandomGenerator(count), m_seed(seed), m_scale(scale)
    {

    }

    float* NormalizedUniformValueGenerator::CreateRandomValues(void)
    {
        srand(m_seed);
        float* rands = new float[m_count];
        for(uint i = 0; i < m_count; ++i)
        {
            float v = rand() / (float)RAND_MAX;
            rands[i] = (2 * v - 1) * m_scale;
        }
        return rands;
    }
}
