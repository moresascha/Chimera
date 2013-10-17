#pragma once
#include "stdafx.h"
#include "ParticleSystem.h"
namespace chimera
{
    class ParticleManager
    {
    private:
        std::list<chimera::ParticleSystem*> m_systems;
    public:
        ParticleManager(VOID);
        VOID Update(ULONG time, UINT dt);
        ~ParticleManager(VOID);
    };
}
