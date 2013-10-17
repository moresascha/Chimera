#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Mat4.h"
#include "Frustum.h"

namespace chimera
{
    class LightComponent;
    class SpotlightNode : public chimera::SceneNode
    {
    private:
        util::ICamera* m_pCamera;
        std::shared_ptr<chimera::LightComponent> m_lightComponent;
        chimera::ShaderProgram* m_drawShadow;
        chimera::ShaderProgram* m_drawShadowInstanced;
        chimera::ShaderProgram* m_drawLighting;
        util::Vec3 m_middle;
        FLOAT m_distance;

    public:
        static chimera::RenderTarget* g_pShadowRenderTarget;
    public:
        SpotlightNode(ActorId actorid);
        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);
        BOOL VIsVisible(SceneGraph* graph);
        VOID VOnRestore(chimera::SceneGraph* graph);
        VOID VOnActorMoved(VOID);
        UINT VGetRenderPaths(VOID);
        ~SpotlightNode(VOID);

        static BOOL Create(VOID);
        static VOID Destroy(VOID);
    };
}
