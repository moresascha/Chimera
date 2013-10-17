#include "ParticleSystem.h"
#include <algorithm>
#include "GameApp.h"
#include "VRamManager.h"
#include "GameView.h"
#include "Geometry.h"
#include "math.h"

namespace chimera
{
    VRamHandle* ParticleQuadGeometryHandleCreator::VGetHandle(VOID)
    {
        return new chimera::Geometry(FALSE);
    }

    VOID ParticleQuadGeometryHandleCreator::VCreateHandle(VRamHandle* handle)
    {

        chimera::Geometry* geo = (chimera::Geometry*)handle;

        FLOAT qscale = 0.01f;

        FLOAT quad[32] = 
        {
            -1 * qscale, -1 * qscale, 0, 0,1,0, 0, 0, 
            +1 * qscale, -1 * qscale, 0, 0,1,0, 1, 0,
            -1 * qscale, +1 * qscale, 0, 0,1,0, 0, 1,
            +1 * qscale, +1 * qscale, 0, 0,1,0, 1, 1
        };

        UINT indices[4] = 
        {
            0, 1, 2, 3
        };

        geo->SetVertexBuffer(quad, 4, 32);
        geo->SetIndexBuffer(indices, 4, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->VCreate();
    }

    std::shared_ptr<chimera::Geometry> ParticleSystem::GetGeometry(VOID)
    {
        if(!m_geometry->VIsReady())
        {
            m_geometry = std::static_pointer_cast<chimera::Geometry>(chimera::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(chimera::VRamResource(".ParticleQuadGeometry")));
        }
        return m_geometry;
    }

    ParticleSystem::ParticleSystem(BaseEmitter* emitter, IRandomGenerator* generator) :
        m_randomValues(NULL), m_pCuda(NULL), m_acceleration(NULL), m_pEmitter(emitter), m_pRandGenerator(generator), m_pParticleBuffer(NULL), m_time(0), m_updateInterval(16) //60 fps
            , m_localWorkSize(256)
    {
    }

    VOID ParticleSystem::AddModifier(BaseModifier* mod)
    {
        m_mods.push_back(mod);
        mod->SetAABB(m_aabb);
    }

    VOID ParticleSystem::RemoveModifier(BaseModifier* mod)
    {
        auto it = std::find(m_mods.begin(), m_mods.end(), mod);
        if(it != m_mods.end())
        {
            BaseModifier* mod = *it;
            m_mods.erase(it);
            SAFE_DELETE(mod);
        }
    }

    VOID ParticleSystem::SetTranslation(CONST util::Vec3& translation)
    {
        m_position = translation;
    }

    CONST util::Vec3& ParticleSystem::GetTranslation(VOID)
    {
        return m_position;
    }

    cudah::cudah* ParticleSystem::GetCuda(VOID)
    {
        return m_pCuda;
    }

    VOID ParticleSystem::UpdateTick(ULONG time, ULONG dt)
    {
        m_pCuda->MapGraphicsResource(m_particles);
        //assert((INT)timer->GetTime() - (INT)m_startTime > 0);
        dt = CLAMP(dt, dt, m_updateInterval);
        m_time += dt;
        FLOAT dtt = (FLOAT)(dt * 1e-3f);
        FLOAT tf = m_time * 1e-3f;
        m_pEmitter->VUpdate(this, tf, dtt);

        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            BaseModifier* mod = *it;
            mod->VUpdate(this, tf, dtt);
        }

        VOID *args[] = { &m_particles->ptr, &m_acceleration->ptr, &m_velocities->ptr, &dtt };

        m_kernel->m_ppArgs = args;

        m_pCuda->CallKernel(m_kernel);
        //integrate((float4*)m_particles->ptr, (float3*)m_acceleration->ptr, (float3*)m_velocities->ptr, (float)(dt * 1e-3f), GetParticlesCount(), GetLocalWorkSize());

        //DEBUG_OUT_A("Count=%d Lws=%u Blocks=%u", GetParticlesCount(), GetLocalWorkSize(), GetParticlesCount() / GetLocalWorkSize());

        m_pCuda->UnmapGraphicsResource(m_particles);
    }

    cudah::cuda_buffer ParticleSystem::GetRandomValues(VOID)
    {
        return m_randomValues;
    }

    cudah::cuda_buffer ParticleSystem::GetAcceleration(VOID)
    {
        return m_acceleration;
    }

    cudah::cuda_buffer ParticleSystem::GetVelocities(VOID)
    {
        return m_velocities;
    }

    cudah::cuda_buffer ParticleSystem::GetParticles(VOID)
    {
        return m_particles;
    }

    UINT ParticleSystem::GetParticlesCount(VOID)
    {
        return m_pEmitter->GetParticleCount();
    }

    BOOL ParticleSystem::HasRandBuffer(VOID)
    {
        return m_randomValues != NULL;
    }

    VOID ParticleSystem::SetAxisAlignedBB(util::AxisAlignedBB& aabb)
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

    util::AxisAlignedBB& ParticleSystem::GetAxisAlignedBB(VOID)
    {
        return m_aabb;
    }

    chimera::VertexBuffer* ParticleSystem::GetParticleBuffer(VOID)
    {
        return m_pParticleBuffer;
    }

    UINT ParticleSystem::GetLocalWorkSize(VOID)
    {
        return m_localWorkSize;
    }

    //vram interface
    BOOL ParticleSystem::VCreate(VOID)
    {
        SAFE_DELETE(m_pCuda);

        m_pCuda = new cudah::cudah("./chimera/ptx/Particles.ptx");

        m_kernel = m_pCuda->GetKernel("_integrate");

        INT blockSize = 256;//todo cudah::cudah::GetMaxThreadsPerSM() / cudah::cudah::GetMaxBlocksPerSM();
        m_localWorkSize = blockSize;

        if(m_pEmitter->GetParticleCount() % m_localWorkSize != 0)
        {
            LOG_CRITICAL_ERROR("particle count not supported");
        }
        
        ParticlePosition* positions = m_pEmitter->CreateParticles();
        FLOAT* rands = NULL;

        if(m_pRandGenerator)
        {
            rands = m_pRandGenerator->CreateRandomValues();
            m_randomValues = m_pCuda->CreateBuffer(std::string("randoms"), m_pRandGenerator->GetValuesCount() * 4, rands, 4);
        }

        FLOAT* parts = (FLOAT*)positions;

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
        for(UINT i = 0; i < m_pEmitter->GetParticleCount(); ++i)
        {
            veloAccInit[i].x = 0;
            veloAccInit[i].y = 0;
            veloAccInit[i].z = 0;
        }

        m_acceleration = m_pCuda->CreateBuffer(std::string("acc"), m_pEmitter->GetParticleCount() * 3 * 4, veloAccInit, 12);

        m_velocities = m_pCuda->CreateBuffer(std::string("velo"), m_pEmitter->GetParticleCount() * 3 * 4, veloAccInit, 12);

        m_geometry = std::static_pointer_cast<chimera::Geometry>(chimera::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(chimera::VRamResource(".ParticleQuadGeometry")));

        //integrate((float4*)m_particles->ptr, (float3*)m_acceleration->ptr, (float3*)m_velocities->ptr, (float)(dt * 1e-3f), GetParticlesCount(), GetLocalWorkSize());
        INT threads = cudahu::GetThreadCount(GetParticlesCount(), GetLocalWorkSize());
        m_kernel->m_blockDim.x = GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        SAFE_ARRAY_DELETE(veloAccInit);
        SAFE_ARRAY_DELETE(rands);
        SAFE_ARRAY_DELETE(positions);

        m_time = 0;

        return TRUE;
    }

    VOID ParticleSystem::VDestroy(VOID)
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

    UINT ParticleSystem::VGetByteCount(VOID) CONST
    {
        //TODO modifier
        //per particle
        //acceleration 3 * 4 byte
        //velocity 3 * 4 byte
        //position 4 * 4 byte
        UINT modBytes = 0;
        TBD_FOR(m_mods)
        {
            modBytes += (*it)->VGetByteCount();
        }
        UINT particles = m_pEmitter->GetParticleCount();
        UINT byte = sizeof(FLOAT);
        return particles * (3 + 3 + 4) * byte + modBytes;
    }

    ParticleSystem::~ParticleSystem(VOID)
    {

    }

    PositiveNormalizedUniformValueGenerator::PositiveNormalizedUniformValueGenerator(UINT count, INT seed) : IRandomGenerator(count), m_seed(seed)
    {

    }

    FLOAT* PositiveNormalizedUniformValueGenerator::CreateRandomValues(VOID)
    {
        srand(m_seed);
        FLOAT* rands = new FLOAT[m_count];
        for(UINT i = 0; i < m_count; ++i)
        {
            rands[i] = rand() / (FLOAT)RAND_MAX;
        }
        return rands;
    }

    NormalizedUniformValueGenerator::NormalizedUniformValueGenerator(UINT count, INT seed, FLOAT scale) : IRandomGenerator(count), m_seed(seed), m_scale(scale)
    {

    }

    FLOAT* NormalizedUniformValueGenerator::CreateRandomValues(VOID)
    {
        srand(m_seed);
        FLOAT* rands = new FLOAT[m_count];
        for(UINT i = 0; i < m_count; ++i)
        {
            float v = rand() / (FLOAT)RAND_MAX;
            rands[i] = (2 * v - 1) * m_scale;
        }
        return rands;
    }
}
