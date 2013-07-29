#pragma once
#include "stdafx.h"
#include "AxisAlignedBB.h"
#include "Timer.h"
#include "cudah.h"
#include "Actor.h"
#include "Event.h"
#include "Resources.h"
#include "VRamManager.h"

namespace d3d
{
    class Geometry;
    class VertexBuffer;
}

namespace tbd
{

    class VRamManager;
    class VRamHandle;
    class IVRamHandleCreator;

    class ParticleQuadGeometryHandleCreator : public IVRamHandleCreator
    {
    public:

        VRamHandle* VGetHandle(VOID);

        VOID VCreateHandle(VRamHandle* handle);
    };

    struct FLOAT4
    {
        FLOAT x;
        FLOAT y;
        FLOAT z;
        FLOAT w;

        FLOAT4(VOID) : x(0), y(0), z(0), w(0)
        {
        }

        FLOAT4(FLOAT x, FLOAT y, FLOAT z, FLOAT w) : x(x), y(y), z(z), w(w)
        {
        }
    };

    typedef FLOAT4 ParticlePosition;

    struct BaseParticleProperties
    {
        FLOAT4 velocity;
    };

    class BaseModifier
    {
        friend class ParticleSystem;
    protected:
        util::AxisAlignedBB m_aabb;
        cudah::cuda_kernel m_kernel;
    public:
        virtual VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt) = 0;

        virtual VOID VOnRestore(ParticleSystem* sys) { }

        virtual UINT VGetByteCount(VOID) { return 0; }

        VOID SetAABB(CONST util::AxisAlignedBB& aabb);

        virtual ~BaseModifier(VOID) {}
    };

    class IRandomGenerator
    {
    protected:
        UINT m_count;
    public:
        IRandomGenerator(UINT count) : m_count(count) {}
        virtual FLOAT* CreateRandomValues(VOID) = 0;
        UINT IRandomGenerator::GetValuesCount(VOID) { return m_count;}
        virtual ~IRandomGenerator(VOID) { }
    };

    class BaseEmitter : public BaseModifier
    {
    protected:
        FLOAT m_startSpawnTime;
        FLOAT m_endSpawnTime;
        UINT m_particleCount;
        cudah::cuda_buffer m_emitterData;
        cudah::cuda_buffer m_startingPositions;
        util::Vec3 m_position;

    public:
        BaseEmitter(CONST util::Vec3& position, UINT count, FLOAT starSpawnTime, FLOAT endSpawnTime);

        virtual VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        virtual VOID VOnRestore(ParticleSystem* sys);

        virtual ParticlePosition* CreateParticles(VOID) = 0;

        UINT GetParticleCount(VOID) CONST { return m_particleCount; }
        FLOAT GetStartSpawnTime(VOID) CONST { return m_startSpawnTime; }
        FLOAT GetEndSpawnTime(VOID) CONST { return m_endSpawnTime; }

        virtual ~BaseEmitter(VOID) {}
    };

    class PositiveNormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        UINT m_seed;
    public:
        PositiveNormalizedUniformValueGenerator(UINT count, INT seed);
        FLOAT* CreateRandomValues(VOID);
    };

    class NormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        UINT m_seed;
        FLOAT m_scale;
    public:
        NormalizedUniformValueGenerator(UINT count, INT seed, FLOAT scale);
        FLOAT* CreateRandomValues(VOID);
    };

    class ParticleSystem : public VRamHandle
    {
    private:
        std::shared_ptr<d3d::Geometry> m_geometry;
        std::vector<BaseModifier*> m_mods;
        util::AxisAlignedBB m_aabb;
        BaseEmitter* m_pEmitter;
        IRandomGenerator* m_pRandGenerator;

        cudah::cuda_buffer m_randomValues;
        cudah::cuda_buffer m_acceleration;
        cudah::cuda_buffer m_velocities;
        cudah::cuda_buffer m_particles;

        cudah::cudah* m_pCuda;

        cudah::cuda_kernel m_kernel;

        d3d::VertexBuffer* m_pParticleBuffer;

        UINT m_updateInterval;
        UINT m_time;

        UINT m_localWorkSize;

        util::Vec3 m_position;

    public:

        ParticleSystem(BaseEmitter* emitter, IRandomGenerator* generator = NULL);

        std::shared_ptr<d3d::Geometry> GetGeometry(VOID);

        VOID AddModifier(BaseModifier* mod);

        VOID RemoveModifier(BaseModifier* mod);
        
        VOID UpdateTick(ULONG time, ULONG dt);

        BOOL HasRandBuffer(VOID);

        UINT GetParticlesCount(VOID);

        UINT GetLocalWorkSize(VOID);

        VOID SetAxisAlignedBB(util::AxisAlignedBB& aabb);

        VOID SetTranslation(CONST util::Vec3& translation);

        CONST util::Vec3& GetTranslation(VOID);

        d3d::VertexBuffer* GetParticleBuffer(VOID);

        CONST util::AxisAlignedBB& GetAxisAlignedBB(VOID);

        cudah::cuda_buffer GetAcceleration(VOID);

        cudah::cuda_buffer GetVelocities(VOID);

        cudah::cuda_buffer GetParticles(VOID);

        cudah::cuda_buffer GetRandomValues(VOID);

        cudah::cudah* GetCuda(VOID);

        //vram interface
        BOOL VCreate(VOID);
        VOID VDestroy();
        UINT VGetByteCount(VOID) CONST;

        ~ParticleSystem(VOID);
    };

    //some Modifier

    class Gravity : public BaseModifier
    {
    private:
        FLOAT m_factor;
        UCHAR m_axis;
    public:
        Gravity(FLOAT factor, UCHAR axis);
        VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        VOID VOnRestore(ParticleSystem* sys);
    };

    class Turbulence : public BaseModifier
    {
    private:
        cudah::cuda_buffer m_randomDirections;
        INT m_seed;
        FLOAT m_strength;
    public:
        Turbulence(INT seed, FLOAT strength) : m_seed(seed), m_strength(strength) {}
        VOID VOnRestore(ParticleSystem* sys);
        VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        UINT VGetByteCount(VOID);
    };

    class ActorBasedModifier : public BaseModifier
    {
    protected:
        ActorId m_actorId;
        std::shared_ptr<tbd::Actor> CreateModActor(CONST util::Vec3& pos, LPCSTR info, CONST FLOAT scale = 1.0f);

    public:
        ActorBasedModifier(VOID);
        VOID ActorMovedDelegate(event::IEventPtr pEventData);
        virtual VOID VOnActorMoved(std::shared_ptr<tbd::Actor> actor) = 0;
        virtual ~ActorBasedModifier(VOID);
    };

    class GradientField : public ActorBasedModifier
    {
    private:
        ActorId m_actorId;
        float4 m_positionNscale;
    public:
        GradientField(VOID);
        VOID VOnRestore(ParticleSystem* sys);
        VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        VOID VOnActorMoved(std::shared_ptr<tbd::Actor> actor);
        UINT VGetByteCount(VOID);
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
        FLOAT m_scale;
        GravityPolarization m_pole;
    public:
        GravityField(CONST util::Vec3& position, CONST FLOAT range, CONST FLOAT scale, GravityPolarization pole);
        VOID VOnRestore(ParticleSystem* sys);
        VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        VOID VOnActorMoved(std::shared_ptr<tbd::Actor> actor);
    };

    class VelocityDamper : public BaseModifier
    {
    private:
        FLOAT m_dampValue;
    public:
        VelocityDamper(FLOAT damping) : m_dampValue(damping) {}
        VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt);
        VOID VOnRestore(ParticleSystem* sys);
    };


    //--emitter

    class PointEmitter : public BaseEmitter
    {
    public:
        PointEmitter(CONST util::Vec3& point, UINT particleCount, FLOAT startSpawn, FLOAT endSpawn);
        ParticlePosition* CreateParticles(VOID);
    };

    class BoxEmitter : public BaseEmitter
    {
    private:
        util::Vec3 m_extends;
    public:
        BoxEmitter(CONST util::Vec3& extends, CONST util::Vec3& point, UINT particleCount, FLOAT startSpawn, FLOAT endSpawn);
        ParticlePosition* CreateParticles(VOID);
    };

    class SurfaceEmitter : public BaseEmitter
    {
    private:
        tbd::Resource m_meshFile;
    public:
        SurfaceEmitter(tbd::Resource meshFile, CONST util::Vec3& position, UINT particleCount, FLOAT start, FLOAT end);
        ParticlePosition* CreateParticles(VOID);
    };
}
