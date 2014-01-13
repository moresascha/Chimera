#include "SceneNode.h"
#include "Mat4.h"
#include "Components.h"
#include "Mesh.h"
#include "Frustum.h"
#include "Material.h"
namespace chimera
{
    VOID DrawActorInfos(IActor* actor, CONST util::Mat4* matrix, ICamera* camera)
    {
        
        /*util::Vec3 pos(matrix->GetTranslation());
        pos = util::Mat4::Transform(camera->GetView(), pos);
        pos = util::Mat4::Transform(camera->GetProjection(), pos);
        pos.x = 0.5 * pos.x + 0.5;
        pos.y = 0.5 * pos.y + 0.5;
        std::stringstream ss;
        ss << "Actor:\n";
        ss << "id=";
        ss << actor->GetId();
        ss << "\n";
        util::Vec3 p(matrix->GetTranslation());
        ss << "pos= (" << p.x << ", " << p.y << ", " << p.z << ")\n";
        app::g_pApp->GetFontManager()->RenderText(ss.str(), pos.x, pos.y); */
    }
    /*
    VOID DrawPicking(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<chimera::Mesh> mesh, std::shared_ptr<chimera::Geometry> geo)
    {
        if(actor->HasComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID))
        {
            chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
            chimera::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            geo->Bind();
            for(auto it = mesh->GetIndexBufferIntervals().begin(); it != mesh->GetIndexBufferIntervals().end(); ++it)
            {
                geo->Draw(it->start, it->count);
            }
        }
    }
    */
    VOID DrawToShadowMap(std::shared_ptr<IGeometry> geo, std::shared_ptr<IMesh> mesh, CONST util::Mat4* matrix)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*matrix);
        geo->VBind();
        for(auto it = mesh->VGetIndexBufferIntervals().begin(); it != mesh->VGetIndexBufferIntervals().end(); ++it)
        {
            geo->VSetTopology(it->topo);
            geo->VDraw(it->start, it->count);
        } 
    }

    MeshNode::MeshNode(ActorId actorid, CMResource ressource) : SceneNode(actorid), m_ressource(ressource), m_longestScale(1)
    {
        VSetRenderPaths(CM_RENDERPATH_ALBEDO | CM_RENDERPATH_SHADOWMAP);
    }

    VOID MeshNode::VOnRestore(ISceneGraph* graph)
    {
        m_pGeometry = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        m_mesh = std::static_pointer_cast<IMesh>(CmGetApp()->VGetCache()->VGetHandle(m_ressource));
        m_materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(m_mesh->VGetMaterials()));
        m_diffuseTexturesCount = 0;

        VOnActorMoved();
        
        std::vector<IndexBufferInterval>& ivals = m_mesh->VGetIndexBufferIntervals();
        for(UINT i = 0; i < ivals.size(); ++i)
        {
            IMaterial* mat = m_materials->GetMaterial(ivals[i].material).get();
            std::shared_ptr<IVRamHandle> handle = CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(mat->VGetTextureDiffuse());
            m_diffuseTextures[m_diffuseTexturesCount] = std::static_pointer_cast<IDeviceTexture>(handle);

            if(mat->VGetTextureNormal().m_name != "unknown")
            {
                handle = CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(mat->VGetTextureNormal());
                m_normalTextures[m_diffuseTexturesCount] = std::static_pointer_cast<IDeviceTexture>(handle);
            }

            m_diffuseTexturesCount++;
        }
    }

    VOID MeshNode::VOnActorMoved(VOID)
    {
        m_transformedBBPoint = util::Mat4::Transform(*VGetTransformation(), m_mesh->VGetAABB().GetMiddle());
        m_longestScale = abs(max(max(VGetTransformation()->GetScale().x, VGetTransformation()->GetScale().y), VGetTransformation()->GetScale().z));
    }

    BOOL MeshNode::VIsVisible(ISceneGraph* graph)
    {
        if(m_pGeometry->VIsReady())
        {
            m_pGeometry->VUpdate();
            if(m_mesh->VIsReady())
            {
                util::AxisAlignedBB& aabb = m_mesh->VGetAABB();
                BOOL in = graph->VGetFrustum()->IsInside(m_transformedBBPoint, m_longestScale * aabb.GetRadius());
                return in;
            }
            else
            {
                VOnRestore(graph);
            }
        }
        else
        {
            m_pGeometry = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        }
        return TRUE;
    }

    VOID MeshNode::_VRender(ISceneGraph* graph, RenderPath& path) 
    {
        //DEBUG_OUT("waiting in scene node for" + m_ressource.m_name);
        //DEBUG_OUT("done");
        if(m_pGeometry->VIsReady())
        {
            m_pGeometry->VUpdate();
            if(m_mesh->VIsReady() && m_materials->VIsReady())
            {
                switch(path)
                {
                case CM_RENDERPATH_SHADOWMAP : 
                    {
                        DrawToShadowMap(m_pGeometry, m_mesh, VGetTransformation());
                        /*app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*m_transformation->GetTransformation());
                        m_pGeometry->Bind();
                        for(auto it = m_mesh->GetIndexBufferIntervals().begin(); it != m_mesh->GetIndexBufferIntervals().end(); ++it)
                        {
                            m_pGeometry->Draw(it->start, it->count);
                        } */
                    } break;
                case CM_RENDERPATH_ALBEDO_WIRE :
                case CM_RENDERPATH_ALBEDO :
                    {
                        DrawToAlbedo();
                    } break;
                /*case eRenderPath_DrawBounding :
                    {
                        DrawSphere(GetTransformation(), m_mesh->GetAABB());
                    } break;

                case eRenderPath_DrawPicking :
                    {
                        DrawPicking(m_actor, GetTransformation(), m_mesh, m_pGeometry);
                    } break;

                case eRenderPath_DrawDebugInfo : 
                    {
                        chimera::DrawActorInfos(m_actor, GetTransformation(), graph->GetCamera());
                    } break;*/
                }
            }
            else
            {
                VOnRestore(graph); 
            }
        }
        else
        {
            m_pGeometry = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        }
    }

    VOID MeshNode::DrawToAlbedo(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*VGetTransformation());
        m_pGeometry->VBind();
        //CmGetApp()->VGetHumanView()->VGetRenderer()->VSetActorId(this->m_actor->GetId());

        UINT texPos = 0;
        for(auto it = m_mesh->VGetIndexBufferIntervals().begin(); it != m_mesh->VGetIndexBufferIntervals().end(); ++it)
        {
            if(m_materials->VIsReady())
            {
                chimera::IMaterial* mat = m_materials->GetMaterial(it->material).get();
                IDeviceTexture* t = m_diffuseTextures[texPos].get();

                if(t && t->VIsReady())
                {
                    t->VUpdate();
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VSetDiffuseTexture(t);
                    IDeviceTexture* norm = m_normalTextures[texPos].get();
                    if(norm)
                    {
                        if(norm->VIsReady())
                        {
                            norm->VUpdate();
                            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetTexture(eNormalColorSampler, norm);
                            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(TRUE);
                        }
                        else
                        {
                            m_normalTextures[texPos] = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(mat->VGetTextureNormal()));
                        }
                    }
                    else
                    {
                        CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(FALSE);
                    }
                }
                else
                {
                    m_diffuseTextures[texPos] = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(mat->VGetTextureDiffuse()));
                    CmGetApp()->VGetHumanView()->VGetRenderer()->VSetDefaultMaterial();
                }
                CmGetApp()->VGetHumanView()->VGetRenderer()->VPushMaterial(*mat);
                m_pGeometry->VSetTopology(it->topo);
                m_pGeometry->VDraw(it->start, it->count);
            }
            else
            {
                m_materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(m_mesh->VGetMaterials()));
            }
            texPos++;
        }
    }

    MeshNode::~MeshNode(VOID)
    {
        
    }
}
