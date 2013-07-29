#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Geometry.h"
#include "Actor.h"
#include "ShaderProgram.h"
#include "ParticleSystem.h"

namespace tbd
{
    class ParticleNode : public SceneNode
    {
    private:
        std::shared_ptr<tbd::ParticleSystem> m_pParticleSystem;
        UINT m_time;
        util::Vec3 m_transformedBBPoint;
        util::Timer m_timer;

    public:
        ParticleNode(ActorId id);

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        BOOL VIsVisible(SceneGraph* graph);

        VOID VOnRestore(tbd::SceneGraph* graph);

        VOID VOnUpdate(ULONG millis, SceneGraph* graph);

        VOID VOnActorMoved(VOID);

        UINT VGetRenderPaths(VOID);

        ~ParticleNode(VOID);
    };
}


