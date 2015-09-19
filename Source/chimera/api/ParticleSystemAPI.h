#pragma once
#include "CMTypes.h"

namespace nutty
{
    class cuKernel;
}

struct ParticlePosition;

namespace chimera
{
    namespace util
    {
        class Plane;
    };

    typedef void* MemPtr;
    typedef ParticlePosition* (*ParticleCreator)(uint count);
    typedef void (*RNG)(uint count, float**);

    class IParticleEmitter;

    class IParticleModifier
    {
    public:
        virtual void VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt) = 0;

        virtual void VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter) = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual void VSetAABB(const util::AxisAlignedBB& aabb) = 0;

        virtual void VRelease(void) = 0;

        virtual ~IParticleModifier(void) { }
    };

    class ICudaParticleModifier : virtual public IParticleModifier
    {
    public:
        virtual nutty::cuKernel* VGetKernel(void) = 0;

        //todo
//         template <
//             typename T
//         >
//         void VSetKernelArg(uint index, T t)
//         {
//             VGetKernel()->SetKernelArg(index, t);
//         }
    };

    class IParticleEmitter : public virtual IParticleModifier
    {
    public:
        virtual uint VGetParticleCount(void) const = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual void VSetParticleCreator(ParticleCreator creator) = 0;

        virtual void VCreateParticles(ParticlePosition* dst) = 0;

        virtual void VSetPosition(const util::Vec3& pos) = 0;

        virtual void VMapArrays(void) = 0;

        virtual void VUnmapArrays(void) = 0;

        virtual MemPtr VGetVelocitiesArray(void) = 0;

        virtual MemPtr VGetAccelerationArray(void) = 0;

        virtual MemPtr VGetParticleArray(void) = 0;

        virtual IVertexBuffer* VGetGFXPosArray(void) = 0;

        virtual IVertexBuffer* VGetGFXVeloArray(void) = 0;

        virtual ~IParticleEmitter(void) { }
    };

    class IParticleSystem
    {
    public:
        virtual IParticleModifier* VAddModifier(std::unique_ptr<IParticleModifier>& mod) = 0;

        virtual IParticleEmitter* VAddEmitter(std::unique_ptr<IParticleEmitter>& emitter) = 0;

        virtual void VAddForce(const util::Vec3* dir) = 0;

        virtual void VAddSpawn(const util::Vec3* pos, uint particles) = 0;

        virtual void VAddDrain(const util::Plane* drain) = 0;

        virtual bool VRemoveEmitter(IParticleEmitter* emitter) = 0;

        virtual bool VRemoveModifier(IParticleModifier* modifier) = 0;

        virtual void VUpdate(ulong time, uint dt) = 0;

        virtual uint VGetParticleCount(void) const = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual std::vector<IParticleEmitter*>& VGetEmitter(void) = 0;

        virtual void VOnRestore(void) = 0;

        virtual void VRelease(void) = 0;

        virtual MemPtr VGetDeviceRNGArray(void) const = 0;

        virtual const util::AxisAlignedBB& VGetAxisAlignedBB(void) const = 0;

        virtual ~IParticleSystem(void) { }
    };

    class IParticleFactory
    {
    public:
        virtual IParticleSystem* VCreateParticleSystem(void) = 0;

        virtual IParticleModifier* VCreateParticleModifier(ICMStream* stream) = 0;

        virtual IParticleModifier* VCreateCudaBaseParticleModifier(const char* module, const char* function) = 0;

        virtual IParticleEmitter* VCreateEmitter(ICMStream* stream) = 0;

        virtual IParticleSystem* VAssemblyParticleSystem(ICMStream* stream) = 0;
    };

    class IParticleManager
    {
    public:
        virtual IParticleSystem* VCreateParticleSystem(ICMStream* stream = NULL) = 0;
    };
}