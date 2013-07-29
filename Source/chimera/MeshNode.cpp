#include "SceneNode.h"
#include "GameApp.h"
#include "Texture.h"
#include "VRamManager.h"
#include "GeometryFactory.h"
#include "D3DRenderer.h"
#include "Mat4.h"
#include "Components.h"
#include "Mesh.h"
#include "SceneGraph.h"
#include "Frustum.h"
namespace tbd
{

    
    VOID DrawActorInfos(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<util::ICamera> camera)
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

    VOID DrawPicking(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<tbd::Mesh> mesh, std::shared_ptr<d3d::Geometry> geo)
    {
        if(actor->HasComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID))
        {
            app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
            app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            geo->Bind();
            for(auto it = mesh->GetIndexBufferIntervals().begin(); it != mesh->GetIndexBufferIntervals().end(); ++it)
            {
                geo->Draw(it->start, it->count);
            }
        }
    }

    VOID DrawToShadowMap(std::shared_ptr<d3d::Geometry> geo, std::shared_ptr<tbd::Mesh> mesh, CONST util::Mat4* matrix)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
        geo->Bind();
        for(auto it = mesh->GetIndexBufferIntervals().begin(); it != mesh->GetIndexBufferIntervals().end(); ++it)
        {
            geo->Draw(it->start, it->count);
        } 
    }

    MeshNode::MeshNode(ActorId actorid, tbd::Resource ressource) : SceneNode(actorid), m_ressource(ressource)
    {
    }

    MeshNode::MeshNode(tbd::Resource ressource) : SceneNode(), m_ressource(ressource)
    {
    }

    VOID MeshNode::VOnRestore(tbd::SceneGraph* graph)
    {
        m_geo = std::static_pointer_cast<d3d::Geometry>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_ressource));
        m_mesh = std::static_pointer_cast<tbd::Mesh>(app::g_pApp->GetCache()->GetHandle(m_ressource));
        m_materials = std::static_pointer_cast<tbd::MaterialSet>(app::g_pApp->GetCache()->GetHandle(m_mesh->GetMaterials()));
        m_diffuseTexturesCount = 0;

        VOnActorMoved();

        for(auto it = m_mesh->GetIndexBufferIntervals().begin(); it != m_mesh->GetIndexBufferIntervals().end(); ++it)
        {
            tbd::IMaterial* mat = m_materials->GetMaterial(it->material).get();
            std::shared_ptr<tbd::VRamHandle> handle = app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(mat->VGetTextureDiffuse());
            m_diffuseTextures[m_diffuseTexturesCount] = std::static_pointer_cast<d3d::Texture2D>(handle);

            if(mat->VGetTextureNormal().m_name != "unknown")
            {
                handle = app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(mat->VGetTextureNormal());
                m_normalTextures[m_diffuseTexturesCount] = std::static_pointer_cast<d3d::Texture2D>(handle);
            }

            m_diffuseTexturesCount++;
        }
    }

    UINT MeshNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO | eDRAW_TO_SHADOW_MAP | eDRAW_BOUNDING_DEBUG | eDRAW_PICKING;
    }

    VOID MeshNode::VOnActorMoved(VOID)
    {
        //if(this->m_transformation)
        {
            m_transformedBBPoint = util::Mat4::Transform(*this->m_transformation->GetTransformation(), m_mesh->GetAABB().GetMiddle());
        }
    }

    BOOL MeshNode::VIsVisible(SceneGraph* graph)
    {
        if(m_geo->IsReady())
        {
            m_geo->Update();
            if(m_mesh->IsReady())
            {
                m_mesh->Update();
                util::AxisAlignedBB& aabb = m_mesh->GetAABB();
                FLOAT scale = m_transformation->GetTransformation()->GetScale().x;
                BOOL in = graph->GetFrustum()->IsInside(m_transformedBBPoint, scale * aabb.GetRadius());
                return in;
            }
            else
            {
                m_mesh = std::static_pointer_cast<tbd::Mesh>(app::g_pApp->GetCache()->GetHandle(m_ressource));
            }
        }
        else
        {
            m_geo = std::static_pointer_cast<d3d::Geometry>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_ressource));
        }
        return FALSE;
    }

    VOID MeshNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path) 
    {
        //DEBUG_OUT("waiting in scene node for" + m_ressource.m_name);
        //DEBUG_OUT("done");
        if(m_geo->IsReady())
        {
            m_geo->Update();
            if(m_mesh->IsReady())
            {
                m_mesh->Update();
                switch(path)
                {
                case eDRAW_TO_SHADOW_MAP: 
                    {
                        DrawToShadowMap(m_geo, m_mesh, m_transformation->GetTransformation());
                        /*app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*m_transformation->GetTransformation());
                        m_geo->Bind();
                        for(auto it = m_mesh->GetIndexBufferIntervals().begin(); it != m_mesh->GetIndexBufferIntervals().end(); ++it)
                        {
                            m_geo->Draw(it->start, it->count);
                        } */
                    } break;
                case eDRAW_TO_ALBEDO :
                    {
                        DrawToAlbedo();
                    } break;
                case eDRAW_BOUNDING_DEBUG :
                    {
                        DrawSphere(m_transformation->GetTransformation(), m_mesh->GetAABB());
                    } break;

                case eDRAW_PICKING :
                    {
                        DrawPicking(m_actor, m_transformation->GetTransformation(), m_mesh, m_geo);
                    } break;

                case eDRAW_DEBUG_INFOS : 
                    {
                        tbd::DrawActorInfos(m_actor, this->m_transformation->GetTransformation(), graph->GetCamera());
                    } break;
                }
            }
            else
            {
                m_mesh = std::static_pointer_cast<tbd::Mesh>(app::g_pApp->GetCache()->GetHandleAsync(m_ressource));
            }
        }
        else
        {
            m_geo = std::static_pointer_cast<d3d::Geometry>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_ressource));
        }
    }

    VOID MeshNode::DrawToAlbedo(VOID)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*m_transformation->GetTransformation());
        m_geo->Bind();
        app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(this->m_actor->GetId());

        UINT texPos = 0;
        for(auto it = m_mesh->GetIndexBufferIntervals().begin(); it != m_mesh->GetIndexBufferIntervals().end(); ++it)
        {
            if(m_materials->IsReady())
            {
                m_materials->Update();
                tbd::IMaterial* mat = m_materials->GetMaterial(it->material).get();
                std::shared_ptr<d3d::Texture2D>& t = m_diffuseTextures[texPos];

                if(t != NULL && t->IsReady())
                {
                    t->Update();
                    ID3D11ShaderResourceView* v = t->GetShaderResourceView();
                    app::g_pApp->GetHumanView()->GetRenderer()->SetDiffuseSampler(v);
                    std::shared_ptr<d3d::Texture2D>& norm = m_normalTextures[texPos];
                    if(norm != NULL)
                    {
                        if(norm->IsReady())
                        {
                            norm->Update();
                            app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eNormalColorSampler, norm->GetShaderResourceView());
                            app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(TRUE);
                        }
                        else
                        {
                            m_normalTextures[texPos] = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(mat->VGetTextureNormal()));
                        }
                    }
                    else
                    {
                        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                    }
                }
                else
                {
                    m_diffuseTextures[texPos] = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(mat->VGetTextureDiffuse()));
                    app::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
                }
                app::g_pApp->GetHumanView()->GetRenderer()->VPushMaterial(*mat);
                m_geo->Draw(it->start, it->count);
            }
            else
            {
                m_materials = std::static_pointer_cast<tbd::MaterialSet>(app::g_pApp->GetCache()->GetHandle(m_mesh->GetMaterials()));
            }
            texPos++;
        }
    }

    MeshNode::~MeshNode(VOID)
    {
        
    }
}
