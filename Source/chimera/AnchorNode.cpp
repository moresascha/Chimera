#include "SceneNode.h"
#include "SceneGraph.h"
#include "GameApp.h"
#include "GameView.h"
#include "D3DRenderer.h"
#include "Components.h"
namespace tbd
{    
    UINT AnchorNode::VGetRenderPaths(VOID)
    {
        return eDRAW_EDIT_MODE | eDRAW_PICKING | eDRAW_DEBUG_INFOS;
    }

    VOID AnchorNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        switch(m_meshType)
        {
        case eBOX : 
            {
                switch(path)
                {
                case eDRAW_PICKING :
                    {
                        util::AxisAlignedBB aabb;
                        aabb.AddPoint(util::Vec3(-1,-1,-1));
                        aabb.AddPoint(util::Vec3(1,1,1));
                        aabb.Construct();
                        DrawPickingCube(m_actor, GetTransformation(), aabb);
                    } break;
                case eDRAW_EDIT_MODE :
                    {
                        if(m_drawMode == eWire)
                        {
                            app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
                        }
                        util::AxisAlignedBB aabb;
                        aabb.AddPoint(util::Vec3(-1,-1,-1));
                        aabb.AddPoint(util::Vec3(1,1,1));
                        aabb.Construct();
                        DrawBox(GetTransformation(), aabb);
                        if(m_drawMode == eWire)
                        {
                            app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
                        }
                    } break;
                }

            } break;
        case eSPHERE :
            {
                switch(path)
                {
                case eDRAW_PICKING :
                    {
                        DrawPickingSphere(m_actor, GetTransformation(), m_radius);
                    } break;
                case eDRAW_EDIT_MODE :
                    {
                        if(m_drawMode == eWire)
                        {
                            app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
                        }
                        DrawAnchorSphere(m_actor, GetTransformation(), m_radius);
                        if(m_drawMode == eWire)
                        {
                            app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
                        }
                    } break;
                case eDRAW_DEBUG_INFOS :
                    {
                        DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), m_info);
                    } break;
                }
            } break; 
        }
    }
}

