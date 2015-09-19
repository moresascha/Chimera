#include "SceneNode.h"
#include "Material.h"

namespace chimera
{
    GeometryNode::GeometryNode(ActorId actor, std::unique_ptr<IGeometry> geo) : SceneNode(actor)
    {
        Material* m = new Material();
        m->m_ambient.x = 1;
        m->m_ambient.y = 1;
        m->m_ambient.z = 1;
        m_pMaterial = std::unique_ptr<IMaterial>(m);
        m_pGeometry = std::move(geo);
        VSetRenderPaths(CM_RENDERPATH_ALBEDO | CM_RENDERPATH_SHADOWMAP);
    }

    void GeometryNode::SetMaterial(const IMaterial& mat)
    {
        *m_pMaterial = mat;
    }

    void GeometryNode::VOnRestore(ISceneGraph* graph)
    {
        m_pGeometry->VSetReady(true);
    }

    void GeometryNode::_VRender(ISceneGraph* graph, RenderPath& path)
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
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VPushMaterial(*m_pMaterial);
                    //CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(false);
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

    GeometryNode::~GeometryNode(void)
    {
        m_pGeometry->VDestroy();
    }
}