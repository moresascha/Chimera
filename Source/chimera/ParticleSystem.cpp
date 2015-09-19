#include "ParticleSystem.h"
#include <Nutty.h>
#include <Copy.h>

#ifdef _DEBUG
#pragma comment(lib, "Nuttyx64Debug.lib")
#else
#pragma comment(lib, "Nuttyx64Release.lib")
#endif

namespace chimera
{
    ParticleSystem::ParticleSystem(uint particleCount, RNG generator) : 
        m_fpRandGenerator(generator), m_time(0), m_updateInterval(16)
    {
    }

    IParticleModifier* ParticleSystem::VAddModifier(std::unique_ptr<IParticleModifier>& mod)
    {
        IParticleModifier* m = mod.release();
        m_mods.push_back(m);
        m->VSetAABB(m_aabb);
        return m;
    }

    bool ParticleSystem::VRemoveModifier(IParticleModifier* mod)
    {
        auto it = std::find(m_mods.begin(), m_mods.end(), mod);
        if(it != m_mods.end())
        {
            IParticleModifier* mod = *it;
            m_mods.erase(it);
            SAFE_DELETE(mod);
            return true;
        }
        return false;
    }

    IParticleEmitter* ParticleSystem::VAddEmitter(std::unique_ptr<IParticleEmitter>& emitter)
    {
        IParticleEmitter* m = emitter.release();
        m_emitter.push_back(m);
        m->VSetAABB(m_aabb);
        return m;
    }

    bool ParticleSystem::VRemoveEmitter(IParticleEmitter* mod)
    {
        auto it = std::find(m_emitter.begin(), m_emitter.end(), mod);
        if(it != m_emitter.end())
        {
            IParticleEmitter* mod = *it;
            m_emitter.erase(it);
            SAFE_DELETE(mod);
            return true;
        }
        return false;
    }

    void ParticleSystem::VUpdate(ulong time, uint dt)
    {

        dt = CLAMP(dt, dt, m_updateInterval);
        m_time += (ulong)dt;
        float dtt = (float)(dt * 1e-3f);
        float ftime = time * 1e-3f;
        //DEBUG_OUT_A("%f %f %d\n", ftime, dtt, m_time);

        for(auto& emitter = m_emitter.begin(); emitter != m_emitter.end(); ++emitter)
        {
            IParticleEmitter* em = *emitter;
            em->VMapArrays();

            em->VUpdate(this, NULL, ftime, dtt);

            for(auto& modifier = m_mods.begin(); modifier != m_mods.end(); ++modifier)
            {
                IParticleModifier* mod = *modifier;
                mod->VUpdate(this, em, m_time * 1e-3f, dtt);
            }

            void* posPtr = em->VGetParticleArray();
            void* accPtr = em->VGetAccelerationArray();
            void* veloPtr = em->VGetVelocitiesArray();
            m_kernel.SetRawKernelArg(0, &posPtr);
            m_kernel.SetRawKernelArg(1, &accPtr);
            m_kernel.SetRawKernelArg(2, &veloPtr);
            m_kernel.SetKernelArg(3, dtt); 
            uint N = em->VGetParticleCount();
            m_kernel.SetKernelArg(4, N); 

            uint grid = nutty::cuda::GetCudaGrid(N, CUDA_GROUP_SIZE);
            m_kernel.SetDimension(grid, CUDA_GROUP_SIZE);

            m_kernel.Call();

            em->VUnmapArrays();
        }

        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize());
    }

    uint ParticleSystem::VGetParticleCount(void) const
    {
        uint count = 0;
        for(auto& emitter = m_emitter.begin(); emitter != m_emitter.end(); ++emitter)
        {
            count += (*emitter)->VGetParticleCount();
        }
        return count;
    }

    const util::AxisAlignedBB& ParticleSystem::VGetAxisAlignedBB(void) const
    {
        return m_aabb;
    }

    void ParticleSystem::VOnRestore(void)
    {
        nutty::Init((ID3D11Device*)CmGetApp()->VGetHumanView()->VGetRenderer()->VGetDevice());

        m_cudaMod.Create(KERNEL_PTX);

        m_kernel.Create(m_cudaMod.GetFunction("_integrate"));
        
        uint rngCount = 2048;
        nutty::HostBuffer<float> rands(rngCount);

        if(m_fpRandGenerator)
        {
            float* ptr = rands.Begin()();
            m_fpRandGenerator(rngCount, &ptr);
            m_randomValues.Resize(rngCount);
        }

        for(auto& emitter = m_emitter.begin(); emitter != m_emitter.end(); ++emitter)
        {
            (*emitter)->VOnRestore(this, NULL);
            for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
            {
                IParticleModifier* mod = *it;
                mod->VOnRestore(this, *emitter);
            }
        }

        m_time = 0;
    }

    uint ParticleSystem::VGetByteCount(void) const
    {
        uint modBytes = 0;
        TBD_FOR(m_mods)
        {
            modBytes += (*it)->VGetByteCount();
        }
        TBD_FOR(m_emitter)
        {
            modBytes += (*it)->VGetByteCount();
        }
        return modBytes;
    }

    void ParticleSystem::VRelease(void)
    {
        for(auto it = m_mods.begin(); it != m_mods.end(); ++it)
        {
            IParticleModifier* mod = *it;
            mod->VRelease();
            SAFE_DELETE(mod);
        }

        for(auto it = m_emitter.begin(); it != m_emitter.end(); ++it)
        {
            IParticleEmitter* emitter = *it;
            emitter->VRelease();
            SAFE_DELETE(emitter);
        }
        m_emitter.clear();
        m_mods.clear();
    }

    ParticleSystem::~ParticleSystem(void)
    {
        VRelease();
    }

    PositiveNormalizedUniformValueGenerator::PositiveNormalizedUniformValueGenerator(uint count, int seed) : IRandomGenerator(count), m_seed(seed)
    {

    }

    void PositiveNormalizedUniformValueGenerator::CreateRandomValues(float* rands)
    {
        srand(m_seed);
        for(uint i = 0; i < m_count; ++i)
        {
            rands[i] = rand() / (float)RAND_MAX;
        }
    }

    NormalizedUniformValueGenerator::NormalizedUniformValueGenerator(uint count, int seed, float scale) : IRandomGenerator(count), m_seed(seed), m_scale(scale)
    {

    }

    void NormalizedUniformValueGenerator::CreateRandomValues(float* rands)
    {
        srand(m_seed);
        for(uint i = 0; i < m_count; ++i)
        {
            float v = rand() / (float)RAND_MAX;
            rands[i] = (2 * v - 1) * m_scale;
        }
    }
}
