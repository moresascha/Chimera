#pragma once
#include "stdafx.h"
#include "ParticleSystem.h"
namespace tbd
{
    class ParticleManager
    {
    private:
        std::list<tbd::ParticleSystem*> m_systems;
    public:
        ParticleManager(VOID);
        VOID Update(ULONG time, UINT dt);
        ~ParticleManager(VOID);
    };
}
