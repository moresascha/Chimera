#include "SceneNode.h"
#include "Geometry.h"
#include "GameApp.h"
#include "Mesh.h"
#include "GeometryFactory.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "Components.h"
#include "Mat4.h"
#include "Vec3.h"
#include "Vec4.h"
#include "SceneGraph.h"
#include "Frustum.h"

namespace chimera
{
    GeometryNode::GeometryNode(GeoCreator geoC) : m_pFuncGeometry(geoC), m_pGeometry(NULL)
    {
        m_pMaterial = new chimera::Material();
    }

    UINT GeometryNode::VGetRenderPaths(VOID)
    {
        return eRenderPath_DrawToAlbedo | eRenderPath_DrawToShadowMap | eRenderPath_DrawBounding | eRenderPath_DrawPicking;
    }

    VOID GeometryNode::SetMaterial(CONST chimera::Material& mat)
    {
        SAFE_DELETE(m_pMaterial);
        m_pMaterial = new chimera::Material(mat);
    }

    VOID GeometryNode::VOnRestore(chimera::SceneGraph* graph)
    {
        if(!m_pGeometry || !m_pGeometry->VIsReady())
        {
            std::stringstream ss;
            ss << "Geometry";
            ss << m_actorId;
            m_pGeometry = std::shared_ptr<chimera::Geometry>(m_pFuncGeometry());
            chimera::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(ss.str(), m_pGeometry);
        }
    }

    VOID GeometryNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        if(m_pGeometry->VIsReady())
        {
            m_pGeometry->Update();
            switch(path)
            {
            case eRenderPath_DrawToShadowMap: 
                {
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    m_pGeometry->Bind();
                    m_pGeometry->Draw();
                } break;
            case eRenderPath_DrawToAlbedo :
                {
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                    chimera::g_pApp->GetRenderer()->SetDefaultTexture();
                    chimera::g_pApp->GetRenderer()->VPushMaterial(*m_pMaterial);
                    chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                    m_pGeometry->Bind();
                    m_pGeometry->Draw();
                } break;
            case eRenderPath_DrawBounding :
                {
                    DrawSphere(GetTransformation(), m_aabb);
                } break;
            case eRenderPath_DrawPicking :
                {
                    //DrawPicking(m_actor, m_transformation->GetTransformation(), m_mesh, m_geo);
                } break;
            case eRenderPath_DrawDebugInfo : 
                {
                    chimera::DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
                } break;
            }
        }
        else
        {
            VOnRestore(graph);
        }
    }

    GeometryNode::~GeometryNode(VOID)
    {
        SAFE_DELETE(m_pMaterial);
    }
}