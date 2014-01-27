#include "SceneNode.h"
#include "Mat4.h"
#include "Components.h"
#include "Mesh.h"
#include "Frustum.h"
#include "Material.h"
namespace chimera
{
    void DrawActorInfos(IActor* actor, const util::Mat4* matrix, ICamera* camera)
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
    
    void DrawPicking(IActor* actor, const util::Mat4* matrix, IMesh* mesh, std::shared_ptr<IGeometry> geo)
    {
        if(actor->VHasComponent(CM_CMP_PICKABLE))
        {
            CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*matrix);
            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetActorId(actor->GetId());
            geo->VBind();
            for(auto it = mesh->VGetIndexBufferIntervals().begin(); it != mesh->VGetIndexBufferIntervals().end(); ++it)
            {
                geo->VDraw(it->start, it->count);
            }
        }
    }
    
    void DrawToShadowMap(std::shared_ptr<IGeometry> geo, IMesh* mesh, const util::Mat4* matrix)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*matrix);
        geo->VBind();
        for(auto it = mesh->VGetIndexBufferIntervals().begin(); it != mesh->VGetIndexBufferIntervals().end(); ++it)
        {
            geo->VSetTopology(it->topo);
            geo->VDraw(it->start, it->count);
        } 
    }

    MeshNode::MeshNode(ActorId actorid, CMResource ressource, std::string meshId) : SceneNode(actorid), m_ressource(ressource), m_longestScale(1), m_meshId(meshId)
    {
        VSetRenderPaths(CM_RENDERPATH_ALBEDO | CM_RENDERPATH_SHADOWMAP);
    }

    void MeshNode::VOnRestore(ISceneGraph* graph)
    {
        m_pGeometry = std::static_pointer_cast<IGeometry>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle((m_meshId != "" ? m_meshId + std::string("$") : "") + m_ressource));
        m_meshSet = std::static_pointer_cast<IMeshSet>(CmGetApp()->VGetCache()->VGetHandle(m_ressource));

        if(m_meshId != "")
        {
            m_mesh = m_meshSet->VGetMesh(m_meshId);
        }
        else
        {
            m_mesh = m_meshSet->VGetMesh(0);
        }

        m_materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(m_mesh->VGetMaterials()));
        m_diffuseTexturesCount = 0;

        VOnActorMoved();
        
        std::vector<IndexBufferInterval>& ivals = m_mesh->VGetIndexBufferIntervals();
        for(uint i = 0; i < ivals.size(); ++i)
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

    void MeshNode::VOnActorMoved(void)
    {
        m_transformedBBPoint = util::Mat4::Transform(*VGetTransformation(), m_mesh->VGetAABB().GetMiddle());
        m_longestScale = abs(max(max(VGetTransformation()->GetScale().x, VGetTransformation()->GetScale().y), VGetTransformation()->GetScale().z));
    }

    bool MeshNode::VIsVisible(ISceneGraph* graph)
    {
        if(m_pGeometry->VIsReady())
        {
            m_pGeometry->VUpdate();
            if(m_meshSet->VIsReady())
            {
                util::AxisAlignedBB& aabb = m_mesh->VGetAABB();
                bool in = graph->VGetFrustum()->IsInside(m_transformedBBPoint, m_longestScale * aabb.GetRadius());
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
        return true;
    }

    void MeshNode::_VRender(ISceneGraph* graph, RenderPath& path) 
    {
        //DEBUG_OUT("waiting in scene node for" + m_ressource.m_name);
        //DEBUG_OUT("done");
        if(m_pGeometry->VIsReady())
        {
            m_pGeometry->VUpdate();
            if(m_meshSet->VIsReady() && m_materials->VIsReady())
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
                case CM_RENDERPATH_PICK :
                    {
                        DrawPicking(m_actor, VGetTransformation(), m_mesh, m_pGeometry);
                    } break;
                /*case eRenderPath_DrawBounding :
                    {
                        DrawSphere(GetTransformation(), m_mesh->GetAABB());
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

    void MeshNode::DrawToAlbedo(void)
    {
        CmGetApp()->VGetHumanView()->VGetRenderer()->VPushWorldTransform(*VGetTransformation());
        m_pGeometry->VBind();
        CmGetApp()->VGetHumanView()->VGetRenderer()->VSetActorId(m_actor->GetId());

        uint texPos = 0;
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
                            CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(true);
                        }
                        else
                        {
                            m_normalTextures[texPos] = std::static_pointer_cast<IDeviceTexture>(CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(mat->VGetTextureNormal()));
                        }
                    }
                    else
                    {
                        CmGetApp()->VGetHumanView()->VGetRenderer()->VSetNormalMapping(false);
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

    MeshNode::~MeshNode(void)
    {
        
    }
}
