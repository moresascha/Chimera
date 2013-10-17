#include "SceneNode.h"
#include "SceneGraph.h"
#include "GameApp.h"
#include "GameView.h"
#include "D3DRenderer.h"
#include "Components.h"
namespace chimera
{    
    UINT AnchorNode::VGetRenderPaths(VOID)
    {
        return eRenderPath_DrawEditMode | eRenderPath_DrawPicking | eRenderPath_DrawDebugInfo;
    }

    VOID AnchorNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        switch(m_meshType)
        {
        case eBOX : 
            {
                switch(path)
                {
                case eRenderPath_DrawPicking :
                    {
                        util::AxisAlignedBB aabb;
                        aabb.AddPoint(util::Vec3(-1,-1,-1));
                        aabb.AddPoint(util::Vec3(1,1,1));
                        aabb.Construct();
                        DrawPickingCube(m_actor, GetTransformation(), aabb);
                    } break;
                case eRenderPath_DrawEditMode :
                    {
                        if(m_drawMode == eFillMode_Wire)
                        {
                            chimera::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(chimera::g_pRasterizerStateWrireframe);
                        }
                        util::AxisAlignedBB aabb;
                        aabb.AddPoint(util::Vec3(-1,-1,-1));
                        aabb.AddPoint(util::Vec3(1,1,1));
                        aabb.Construct();
                        DrawBox(GetTransformation(), aabb);
                        if(m_drawMode == eFillMode_Wire)
                        {
                            chimera::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
                        }
                    } break;
                }

            } break;
        case eSPHERE :
            {
                switch(path)
                {
                case eRenderPath_DrawPicking :
                    {
                        DrawPickingSphere(m_actor, GetTransformation(), m_radius);
                    } break;
                case eRenderPath_DrawEditMode :
                    {
                        if(m_drawMode == eFillMode_Wire)
                        {
                            chimera::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(chimera::g_pRasterizerStateWrireframe);
                        }
                        DrawAnchorSphere(m_actor, GetTransformation(), m_radius);
                        if(m_drawMode == eFillMode_Wire)
                        {
                            chimera::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
                        }
                    } break;
                case eRenderPath_DrawDebugInfo :
                    {
                        DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), m_info);
                    } break;
                }
            } break; 
        }
    }
}

