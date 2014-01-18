#pragma once
#include "stdafx.h"
#include "AxisAlignedBB.h"
#include <DeviceBuffer.h>
#include <cuda/Kernel.h>
#include <interop/D3DInterop.h>
#include <interop/shared_resources.h>

namespace chimera
{
    class ParticleQuadGeometryHandleCreator : public IVRamHandleCreator
    {
    public:

        VRamHandle* VGetHandle(void);

        void VCreateHandle(VRamHandle* handle);
    };

    struct FLOAT4
    {
        float x;
        float y;
        float z;
        float w;

        FLOAT4(void) : x(0), y(0), z(0), w(0)
        {
        }

        FLOAT4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
        {
        }
    };

    struct BaseParticleProperties
    {
        FLOAT4 velocity;
    };

    class BaseModifier : public IParticleModifier
    {
        friend class ParticleSystem;
    protected:
        util::AxisAlignedBB m_aabb;
        nutty::cuKernel m_kernel;
        ParticleSystem* m_sys;
    public:

        BaseModifier(void) : m_sys(NULL), m_kernel(NULL) { }

        virtual void VUpdate(ParticleSystem* sys, float time, float dt) = 0;

        virtual void VOnRestore(ParticleSystem* sys) { }

        virtual uint VGetByteCount(void) = 0;

        void VSetAABB(const util::AxisAlignedBB& aabb);

        virtual ~BaseModifier(void) {}
    };

    class BaseEmitter : public BaseModifier
    {
    protected:
        float m_startSpawnTime;
        float m_endSpawnTime;
        uint m_particleCount;
        nutty::DeviceBuffer<float3> m_emitterData;
        nutty::DeviceBuffer<float3> m_startingPositions;
        util::Vec3 m_position;

    public:
        BaseEmitter(const util::Vec3& position, uint count, float starSpawnTime, float endSpawnTime);

        virtual void VUpdate(ParticleSystem* sys, float time, float dt);
        virtual void VOnRestore(ParticleSystem* sys);

        virtual ParticlePosition CreateParticles(void) = 0;

        uint GetParticleCount(void) const { return m_particleCount; }
        float GetStartSpawnTime(void) const { return m_startSpawnTime; }
        float GetEndSpawnTime(void) const { return m_endSpawnTime; }

        virtual ~BaseEmitter(void) {}
    };

    class IRandomGenerator
    {
    protected:
        uint m_count;
    public:
        IRandomGenerator(uint count) : m_count(count) {}
        virtual float* CreateRandomValues(void) = 0;
        uint IRandomGenerator::GetValuesCount(void) { return m_count;}
        virtual ~IRandomGenerator(void) { }
    };

    class PositiveNormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        uint m_seed;
    public:
        PositiveNormalizedUniformValueGenerator(uint count, int seed);
        float* CreateRandomValues(void);
    };

    class NormalizedUniformValueGenerator : public IRandomGenerator
    {
    private:
        uint m_seed;
        float m_scale;
    public:
        NormalizedUniformValueGenerator(uint count, int seed, float scale);
        float* CreateRandomValues(void);
    };

    class ParticleSystem : public VRamHandle, public IParticleSystem
    {
    private:
        std::shared_ptr<IGeometry> m_geometry;
        IVertexBuffer* m_pParticleBuffer;

        std::vector<std::unique_ptr<IParticleModifier>> m_mods;
        util::AxisAlignedBB m_aabb;
        RNG m_fpRandGenerator;

        nutty::DeviceBuffer<float> m_randomValues;
        nutty::DeviceBuffer<float3> m_acceleration;
        nutty::DeviceBuffer<float3> m_velocities;
        nutty::MappedBufferPtr<float4> m_particles;

        nutty::cuKernel m_kernel;

        uint m_updateInterval;
        uint m_time;

        uint m_localWorkSize;

        ParticleCreator m_fpCreator;

    public:
        ParticleSystem(uint particleCount, IRandomGenerator* generator = NULL);

        IGeometry* VGetParticleGeometry(void);

        void VSetParticleCreator(ParticleCreator creator);

        IParticleModifier* VAddModifier(std::unique_ptr<IParticleModifier>& mod);

        bool VRemoveModifier(IParticleModifier* mod);
        
        void VUpdateTick(ulong time, float dt);

        uint VGetParticlesCount(void) const;

        IVertexBuffer* VGetParticleArray(void) const;

        const util::AxisAlignedBB& GetAxisAlignedBB(void) const;

        LPCGPUDEVMEM VGetDeviceVelocitiesArray(void) const;

        LPCGPUDEVMEM VGetDeviceAccelerationArray(void) const;

        LPCGPUDEVMEM VGetDeviceParticlesArray(void) const;

        LPCGPUDEVMEM VGetDeviceRNGArray(void) const;

        //vram interface
        bool VCreate(void);

        void VDestroy();

        uint VGetByteCount(void) const;

        ~ParticleSystem(void);
    };

    //some Modifier

    class Gravity : public BaseModifier
    {
    private:
        float m_factor;
        UCHAR m_axis;
    public:
        Gravity(float factor, UCHAR axis);
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnRestore(ParticleSystem* sys);
    };

    class Turbulence : public BaseModifier
    {
    private:
        nutty::DeviceBuffer<float3> m_randomDirections;
        int m_seed;
        float m_strength;
    public:
        Turbulence(int seed, float strength) : m_seed(seed), m_strength(strength) {}
        void VOnRestore(ParticleSystem* sys);
        void VUpdate(ParticleSystem* sys, float time, float dt);
        uint VGetByteCount(void);
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

    class GradientField : public ActorBasedModifier
    {
    private:
        ActorId m_actorId;
        float4 m_positionNscale;
		//cudah::cuda_array m_array;

    public:
        GradientField(void);
        void VOnRestore(ParticleSystem* sys);
        void VUpdate(ParticleSystem* sys, float time, float dt);
        void VOnActorMoved(IActor* actor);
        uint VGetByteCount(void);
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
