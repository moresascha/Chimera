#include "ParticleManager.h"
#include "Cudah.h"

namespace chimera
{
    ParticleManager::ParticleManager(void)
    {
    }

    void ParticleManager::Update(ulong time, uint dt)
    {
        TBD_FOR(m_systems)
        {
            (*it)->UpdateTick(time, dt);
        }
    }

    ParticleManager::~ParticleManager(void)
    {
    }
}

