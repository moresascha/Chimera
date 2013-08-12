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

namespace tbd
{
    GeometryNode::GeometryNode(d3d::Geometry* geo) : m_pGeometry(geo), m_pMaterial(NULL)
    {
        m_pMaterial = new Material();
    }

    UINT GeometryNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO | eDRAW_TO_SHADOW_MAP | eDRAW_BOUNDING_DEBUG | eDRAW_PICKING;
    }

    VOID GeometryNode::SetMaterial(CONST tbd::Material* mat)
    {
        *m_pMaterial = *mat;
    }

    VOID GeometryNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        switch(path)
        {
        case eDRAW_TO_SHADOW_MAP: 
            {
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                m_pGeometry->Bind();
                m_pGeometry->Draw();
            } break;
        case eDRAW_TO_ALBEDO :
            {
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                app::g_pApp->GetRenderer()->SetDefaultTexture();
                app::g_pApp->GetRenderer()->VPushMaterial(*m_pMaterial);
                app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                m_pGeometry->Bind();
                m_pGeometry->Draw();
            } break;
        case eDRAW_BOUNDING_DEBUG :
            {
                DrawSphere(GetTransformation(), m_aabb);
            } break;
        case eDRAW_PICKING :
            {
                //DrawPicking(m_actor, m_transformation->GetTransformation(), m_mesh, m_geo);
            } break;
        case eDRAW_DEBUG_INFOS : 
            {
                tbd::DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
            } break;
        }
    }

    GeometryNode::~GeometryNode(VOID)
    {
        SAFE_DELETE(m_pMaterial);
        m_pGeometry->VDestroy();
        SAFE_DELETE(m_pGeometry);
    }
}