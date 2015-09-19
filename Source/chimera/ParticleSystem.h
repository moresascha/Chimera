#pragma once
#include "stdafx.h"
#include "AxisAlignedBB.h"
#include <Nutty.h>
#include <cuda/cuda_helper.h>
#include <Copy.h>
#include <DeviceBuffer.h>
#include <cuda/Kernel.h>
#include <cuda/Module.h>
#include <interop/D3DInterop.h>
#include <interop/shared_resources.h>
#include "ParticleData.h"

#define CUDA_GROUP_SIZE (uint)256

#define KERNEL_PTX "../Assets/kernel/Particles.ptx"

namespace chimera
{
    class BaseEmitter : public IParticleEmitter
    {
    protected:
        float m_startSpawnTime;
        float m_endSpawnTime;
        uint m_particleCount;
        util::Vec3 m_pos;

        nutty::DeviceBuffer<EmitterData> m_emitterData;
        nutty::DeviceBuffer<ParticlePosition> m_startingPositions;
        nutty::DeviceBuffer<float3> m_acceleration;

        nutty::DevicePtr<ParticlePosition> m_mappedPositionPtr;
        //nutty::MappedBufferPtr<ParticlePosition> m_mappedPositionResource;
        cudaGraphicsResource_t m_positionResoure;

        nutty::DevicePtr<float3> m_mappedVeloPtr;
        //nutty::MappedBufferPtr<float3> m_mappedVeloResource;
        cudaGraphicsResource_t m_veloResoure;

        nutty::cuKernel m_emitKernel;
        nutty::cuModule m_module;

        ParticleCreator m_fpCreator;

        IVertexBuffer* m_pParticleBuffer;
        IVertexBuffer* m_pParticleVeloBuffer;

    public:
        BaseEmitter(uint count, float starSpawnTime, float endSpawnTime);

        virtual void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt);

        virtual void VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter);

        virtual void VCreateParticles(ParticlePosition* positions);

        uint VGetParticleCount(void) const { return m_particleCount; }

        void VSetPosition(const util::Vec3& pos) { m_pos = pos; }

        void VSetParticleCreator(ParticleCreator creator) { m_fpCreator = creator; }

        void VMapArrays(void);

        void VUnmapArrays(void);

        void VRelease(void);

        MemPtr VGetVelocitiesArray(void);

        MemPtr VGetAccelerationArray(void);

        MemPtr VGetParticleArray(void);

        IVertexBuffer* VGetGFXPosArray(void);

        IVertexBuffer* VGetGFXVeloArray(void);

        uint VGetByteCount(void) const;

        void VSetAABB(const util::AxisAlignedBB& aabb) { }

        virtual ~BaseEmitter(void);
    };

    class BaseModifier : public IParticleModifier
    {
    protected:
        util::AxisAlignedBB m_aabb;
        nutty::cuKernel m_kernel;
        nutty::cuModule m_module;

    public:
        BaseModifier(void) {}

        virtual void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt) = 0;

        virtual void VOnRestore(IParticleSystem* sys) { }

        virtual uint VGetByteCount(void) const = 0;

        void VSetAABB(const util::AxisAlignedBB& aabb) { m_aabb = aabb; }

        virtual void VRelease(void) { }

        virtual ~BaseModifier(void) {}
    };

    class IRandomGenerator
    {
    protected:
        uint m_count;
    public:
        IRandomGenerator(uint count) : m_count(count) {}
        virtual void CreateRandomValues(float* data) = 0;
        uint IRandomGenerator::GetValuesCount(void) { return m_count;}
        virtual ~IRandomGenerator(void) { }
    };

    class PositiveNormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        uint m_seed;
    public:
        PositiveNormalizedUniformValueGenerator(uint count, int seed);
        void CreateRandomValues(float* data);
    };

    class NormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        uint m_seed;
        float m_scale;
    public:
        NormalizedUniformValueGenerator(uint count, int seed, float scale);
        void CreateRandomValues(float* data);
    };

    class ParticleSystem : public IParticleSystem
    {
    private:
        std::vector<IParticleModifier*> m_mods;
        std::vector<IParticleEmitter*> m_emitter;

        util::AxisAlignedBB m_aabb;
        RNG m_fpRandGenerator;

        nutty::DeviceBuffer<float> m_randomValues;

        nutty::cuKernel m_kernel;
        nutty::cuModule m_cudaMod;

        uint m_updateInterval;
        uint m_time;

    public:
        ParticleSystem(uint particleCount, RNG rng = NULL);

        IParticleModifier* VAddModifier(std::unique_ptr<IParticleModifier>& mod);

        IParticleEmitter* VAddEmitter(std::unique_ptr<IParticleEmitter>& emitter);

        std::vector<IParticleEmitter*>& VGetEmitter(void) { return m_emitter; }

        bool VRemoveModifier(IParticleModifier* mod);

        void VAddForce(const util::Vec3* dir) {}

        void VAddDrain(const util::Plane* drain) {}

        void VAddSpawn(const util::Vec3* pos, uint particles) {}

        MemPtr VGetDeviceRNGArray(void) const
        {
            return NULL;
        }

        bool VRemoveEmitter(IParticleEmitter* mod);
        
        void VUpdate(ulong time, uint dt);

        void VRelease(void);

        uint VGetParticleCount(void) const;

        const util::AxisAlignedBB& VGetAxisAlignedBB(void) const;

        void VOnRestore(void);

        uint VGetByteCount(void) const;

        ~ParticleSystem(void);
    };

    //some Modifier

    class Gravity : public BaseModifier
    {
    private:
        util::Vec3 m_dir;

    public:
        Gravity(const util::Vec3& dir);

        void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt);

        void VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter);

        uint VGetByteCount(void) const { return 0; }
    };

    class Turbulence : public BaseModifier
    {
    private:
        nutty::DeviceBuffer<float3> m_randomDirections;
        int m_seed;
        float m_strength;

    public:
        Turbulence(int seed, float strength) : m_seed(seed), m_strength(strength) {}

        void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt);

        void VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter);

        uint VGetByteCount(void) const;
    };

    class ActorBasedModifier : public BaseModifier
    {
    protected:
        ActorId m_actorId;
        IActor* CreateModActor(const util::Vec3& pos, LPCSTR info, const float scale = 1.0f);

    public:
        ActorBasedModifier(void);
        void ActorMovedDelegate(chimera::IEventPtr pEventData);
        virtual void VOnActorMoved(IActor* actor) = 0;
        virtual ~ActorBasedModifier(void);
    };

    class GradientField : public BaseModifier
    {
    private:
        ActorId m_actorId;
        float4 m_positionNscale;
        nutty::DeviceBuffer<float3> m_randomDirections;
        CUarray m_pArray;

    public:
        GradientField(void);

        void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt);

        void VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter);

        uint VGetByteCount(void) const;

        void OnResourceChangedDelegate(IEventPtr event);

        ~GradientField(void);
    };

    enum GravityPolarization
    {
        eAttract,
        eRepel
    };

    class GravityField : public ActorBasedModifier
    {
    private:
        float4 m_posistionNrange;
        ActorId m_actorId;
        float m_scale;
        GravityPolarization m_pole;
    public:
        GravityField(const util::Vec3& position, const float range, const float scale, GravityPolarization pole);
        void VOnRestore(ParticleSystem* sys);
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnActorMoved(IActor* actor);
    };

    class VelocityDamper : public BaseModifier
    {
    private:
        float m_dampValue;
    public:
        VelocityDamper(float damping) : m_dampValue(damping) {}
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnRestore(ParticleSystem* sys);
    };

    class Plane : public BaseModifier 
    {
    private:
        util::Plane m_plane;
    public:
        Plane(const util::Plane& p) : m_plane(p) {}
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnRestore(ParticleSystem* sys);
        uint VGetByteCount(void) { return 0; }
    };

    class BoundingBox : public BaseModifier
    {
    private:
        nutty::DeviceBuffer<float3> m_min;
        nutty::DeviceBuffer<float3> m_max;
        nutty::cuKernel m_second;
        float* m_pData[2];
        util::AxisAlignedBB m_aabb;
    public:
        BoundingBox(void);
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnRestore(ParticleSystem* sys);
        uint VGetByteCount(void);
        const util::AxisAlignedBB& GetAABB(void) { return m_aabb; }
        ~BoundingBox(void);
    };

    //--emitter

    /*
    class PointEmitter : public BaseEmitter
    {
    public:
        PointEmitter(CONST util::Vec3& point, UINT particleCount, FLOAT startSpawn, FLOAT endSpawn);
        ParticlePosition* CreateParticles(VOID);
        UINT VGetByteCount(VOID) { return 0; }
    };

    class BoxEmitter : public BaseEmitter
    {
    private:
        util::Vec3 m_extends;
    public:
        BoxEmitter(CONST util::Vec3& extends, CONST util::Vec3& point, UINT particleCount, FLOAT startSpawn, FLOAT endSpawn);
        ParticlePosition* CreateParticles(VOID);
        UINT VGetByteCount(VOID) { return 0; }
    };

    class SurfaceEmitter : public BaseEmitter
    {
    private:
        chimera::CMResource m_meshFile;
    public:
        SurfaceEmitter(chimera::CMResource meshFile, CONST util::Vec3& position, UINT particleCount, FLOAT start, FLOAT end);
        ParticlePosition* CreateParticles(VOID);
        UINT VGetByteCount(VOID) { return 0; }
    };*/
}
