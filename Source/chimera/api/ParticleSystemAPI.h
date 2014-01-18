#pragma once
#include "CMTypes.h"

namespace chimera
{
    typedef const void* LPCGPUDEVMEM;
    typedef const void* ParticlePosition;
    typedef ParticlePosition* (*ParticleCreator)(uint count);
    typedef float* (*RNG)(uint count);

    class IParticleModifier
    {
    public:
        virtual void VUpdate(IParticleSystem* sys, ulong time, float dt) = 0;

        virtual void VOnRestore(IParticleSystem* sys) = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual void VSetAABB(const util::AxisAlignedBB& aabb) = 0;
    };

    class IParticleSystem
    {
    public:
        virtual IParticleModifier* VAddModifier(std::unique_ptr<IParticleModifier>& mod) = 0;

        virtual void VSetParticleCreator(ParticleCreator creator) = 0;

        virtual bool VRemoveModifier(IParticleModifier* mod) = 0;

        virtual void VUpdate(ulong time, float dt) = 0;

        virtual uint VGetParticleCount(void) const = 0;

        virtual void VOnRestore(void) = 0;

        virtual const util::AxisAlignedBB& GetAxisAlignedBB(void) const = 0;

        virtual LPCGPUDEVMEM VGetDeviceVelocitiesArray(void) const = 0;

        virtual LPCGPUDEVMEM VGetDeviceAccelerationArray(void) const = 0;

        virtual LPCGPUDEVMEM VGetDeviceParticlesArray(void) const = 0;

        virtual LPCGPUDEVMEM VGetDeviceRNGArray(void) const = 0;

        virtual const IGeometry* VGetParticleGeometry(void) const = 0;

        virtual const IVertexBuffer* VGetParticleArray(void) = 0;
    };
}