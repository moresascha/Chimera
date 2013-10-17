#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Mat4.h"
#include "Frustum.h"
namespace chimera
{
    class ShaderProgram;
    class RenderTarget;
}
namespace chimera
{
    class LightComponent;
    class SceneGraph;
}
namespace chimera 
{
    class PointlightNode : public chimera::SceneNode
    {
    private:
        util::Mat4 m_mats[6];
        util::Mat4 m_projection;
        //d3d::RenderTarget* m_cubeMapRenderTarget;
        std::shared_ptr<chimera::LightComponent> m_lightComponent;

        chimera::ShaderProgram* m_drawShadow;
        chimera::ShaderProgram* m_drawShadowInstanced;
        chimera::ShaderProgram* m_drawLighting;

        static chimera::PointLightFrustum g_frustum;
    public:
        static chimera::RenderTarget* g_pCubeMapRenderTarget;
    public:
        PointlightNode(ActorId actorid);
        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);
        BOOL VIsVisible(SceneGraph* graph);
        static BOOL Create(VOID);
        static VOID Destroy(VOID);
        VOID VOnRestore(chimera::SceneGraph* graph);
        VOID VOnActorMoved(VOID);
        UINT VGetRenderPaths(VOID);
        ~PointlightNode(VOID);
    };
};