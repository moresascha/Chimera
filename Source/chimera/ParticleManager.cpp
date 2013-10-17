#include "ParticleManager.h"
#include "Cudah.h"

namespace chimera
{
    ParticleManager::ParticleManager(VOID)
    {
    }

    VOID ParticleManager::Update(ULONG time, UINT dt)
    {
        TBD_FOR(m_systems)
        {
            (*it)->UpdateTick(time, dt);
        }
    }

    ParticleManager::~ParticleManager(VOID)
    {
    }
}

