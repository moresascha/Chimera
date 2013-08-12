#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Mat4.h"
#include "Frustum.h"

namespace tbd
{
    class LightComponent;
    class SpotlightNode : public tbd::SceneNode
    {
    private:
        util::ICamera* m_pCamera;
        std::shared_ptr<tbd::LightComponent> m_lightComponent;
        d3d::ShaderProgram* m_drawShadow;
        d3d::ShaderProgram* m_drawShadowInstanced;
        d3d::ShaderProgram* m_drawLighting;
        util::Vec3 m_middle;
        FLOAT m_distance;

    public:
        static d3d::RenderTarget* g_pShadowRenderTarget;
    public:
        SpotlightNode(ActorId actorid);
        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);
        BOOL VIsVisible(SceneGraph* graph);
        VOID VOnRestore(tbd::SceneGraph* graph);
        VOID VOnActorMoved(VOID);
        UINT VGetRenderPaths(VOID);
        ~SpotlightNode(VOID);

        static BOOL Create(VOID);
        static VOID Destroy(VOID);
    };
}
