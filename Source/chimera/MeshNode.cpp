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
            geo->VDraw(it->start, it->count);
        } 
    }

    MeshNode::MeshNode(ActorId actorid, CMResource ressource) : SceneNode(actorid), m_ressource(ressource)
    {
        VSetRenderPaths(CM_RENDERPATH_ALBEDO | CM_RENDERPATH_SHADOWMAP);
    }

    VOID MeshNode::VOnRestore(ISceneGraph* graph)
    {
        m_geo = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        m_mesh = std::static_pointer_cast<IMesh>(CmGetApp()->VGetCache()->VGetHandle(m_ressource));
        m_materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(m_mesh->VGetMaterials()));
        m_diffuseTexturesCount = 0;

        VOnActorMoved();

        for(auto it = m_mesh->VGetIndexBufferIntervals().begin(); it != m_mesh->VGetIndexBufferIntervals().end(); ++it)
        {
            IMaterial* mat = m_materials->GetMaterial(it->material).get();
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
        //if(this->m_transformation)
        {
            m_transformedBBPoint = util::Mat4::Transform(*VGetTransformation(), m_mesh->VGetAABB().GetMiddle());
        }
    }

    BOOL MeshNode::VIsVisible(ISceneGraph* graph)
    {
        return TRUE;
        if(m_geo->VIsReady())
        {
            m_geo->VUpdate();
            if(m_mesh->VIsReady())
            {
                util::AxisAlignedBB& aabb = m_mesh->VGetAABB();
                FLOAT scale = VGetTransformation()->GetScale().x;
                BOOL in = graph->VGetFrustum()->IsInside(m_transformedBBPoint, scale * aabb.GetRadius());
                return in;
            }
            else
            {
                VOnRestore(graph);
            }
        }
        else
        {
            m_geo = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        }
        return TRUE;
    }

    VOID MeshNode::_VRender(ISceneGraph* graph, RenderPath& path) 
    {
        //DEBUG_OUT("waiting in scene node for" + m_ressource.m_name);
        //DEBUG_OUT("done");
        if(m_geo->VIsReady())
        {
            m_geo->VUpdate();
            if(m_mesh->VIsReady() && m_materials->VIsReady())
            {
                switch(path)
                {
                case CM_RENDERPATH_SHADOWMAP: 
                    {
                        DrawToShadowMap(m_geo, m_mesh, VGetTransformation());
                        /*app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*m_transformation->GetTransformation());
                        m_geo->Bind();
                        for(auto it = m_mesh->GetIndexBufferIntervals().begin(); it != m_mesh->GetIndexBufferIntervals().end(); ++it)
                        {
                            m_geo->Draw(it->start, it->count);
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
                        DrawPicking(m_actor, GetTransformation(), m_mesh, m_geo);
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
            m_geo = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(m_ressource));
        }
    }

    VOID MeshNode::DrawToAlbedo(VOID)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*VGetTransformation());
        m_geo->VBind();
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
                m_geo->VDraw(it->start, it->count);
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
