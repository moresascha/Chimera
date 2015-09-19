#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Timer.h"

namespace chimera
{
    class ParticleNode : public SceneNode
    {
    private:
        IParticleSystem* m_pParticleSystem;
        UINT m_time;
        util::Vec3 m_transformedBBPoint;
        util::Timer m_timer;
        std::shared_ptr<IGeometry> m_pGeometry;
        std::shared_ptr<IVRamHandle> m_particleSystemHandle;

        void OnFileChanged(void);

    public:
        ParticleNode(ActorId id, IParticleSystem* pSystem);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        bool VIsVisible(ISceneGraph* graph);

        void VOnRestore(ISceneGraph* graph);

        void VOnUpdate(ULONG millis, ISceneGraph* graph);

        void VOnActorMoved(void);

        ~ParticleNode(void);
    };
}


