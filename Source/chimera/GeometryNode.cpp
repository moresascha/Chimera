#include "SceneNode.h"
#include "Material.h"

namespace chimera
{
    GeometryNode::GeometryNode(ActorId actor, std::unique_ptr<IGeometry> geo) : SceneNode(actor)
    {
        m_pMaterial = std::unique_ptr<IMaterial>(new Material());
        m_pGeometry = std::move(geo);
        VSetRenderPaths(CM_RENDERPATH_ALBEDO | CM_RENDERPATH_SHADOWMAP);
    }

    VOID GeometryNode::SetMaterial(CONST IMaterial& mat)
    {
        *m_pMaterial = mat;
    }

    VOID GeometryNode::VOnRestore(ISceneGraph* graph)
    {
        m_pGeometry->VSetReady(TRUE);
    }

    VOID GeometryNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        if(m_pGeometry->VIsReady())
        {
            switch(path)
            {
            case CM_RENDERPATH_SHADOWMAP: 
                {
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*VGetTransformation());
                    m_pGeometry->VBind();
                    m_pGeometry->VDraw();
                } break;
            case CM_RENDERPATH_ALBEDO_WIRE :
            case CM_RENDERPATH_ALBEDO :
                {
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*VGetTransformation());
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VSetDefaultTexture();
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VSetDefaultMaterial();
                    //CmGetApp()->VGetHumanView()->VGetRenderer()->VPushMaterial(*m_pMaterial);
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(FALSE);
                    m_pGeometry->VBind();
                    m_pGeometry->VDraw();
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
        m_pGeometry->VDestroy();
    }
}