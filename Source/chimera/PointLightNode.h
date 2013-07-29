#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Mat4.h"
#include "Frustum.h"
namespace d3d
{
    class ShaderProgram;
    class RenderTarget;
}
namespace tbd
{
    class LightComponent;
    class SceneGraph;
}
namespace tbd 
{
    class PointLightNode : public tbd::SceneNode
    {
    private:
        util::Mat4 m_mats[6];
        util::Mat4 m_projection;
        //d3d::RenderTarget* m_cubeMapRenderTarget;
        std::shared_ptr<tbd::LightComponent> m_lightComponent;

        d3d::ShaderProgram* m_drawShadow;
        d3d::ShaderProgram* m_drawShadowInstanced;
        d3d::ShaderProgram* m_drawLighting;

        static tbd::PointLightFrustum g_frustum;
    public:
        static d3d::RenderTarget* g_pCubeMapRenderTarget;
    public:
        PointLightNode(ActorId actorid);
        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);
        BOOL VIsVisible(SceneGraph* graph);
        static BOOL Create(VOID);
        static VOID Destroy(VOID);
        VOID VOnRestore(tbd::SceneGraph* graph);
        VOID VOnActorMoved(VOID);
        UINT VGetRenderPaths(VOID);
        ~PointLightNode(VOID);
    };
};