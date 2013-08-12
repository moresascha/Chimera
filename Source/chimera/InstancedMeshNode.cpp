#include "SceneNode.h"
#include "GameApp.h"
#include "Mesh.h"
#include "Geometry.h"
#include "Components.h"
#include "SceneGraph.h"
#include "Frustum.h"
namespace tbd
{
    InstancedMeshNode::InstancedMeshNode(ActorId actorid, tbd::Resource ressource) : MeshNode(actorid, ressource), m_pInstanceHandle(NULL)
    {
    }

    VOID InstancedMeshNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        if(m_pInstanceHandle->IsReady())
        {
            m_pInstanceHandle->Update();
            if(m_geo->IsReady())
            {
                m_geo->Update();
                switch(path)
                {
                case eDRAW_TO_ALBEDO_INSTANCED :
                    {
                        m_geo->SetInstanceBuffer(m_pInstanceHandle->GetBuffer());
                        DrawToAlbedo();
                        m_geo->SetInstanceBuffer(NULL);
                    } break;
                case eDRAW_TO_SHADOW_MAP_INSTANCED : 
                    {
                        m_geo->SetInstanceBuffer(m_pInstanceHandle->GetBuffer());
                        DrawToShadowMap(m_geo, m_mesh, GetTransformation());
                        m_geo->SetInstanceBuffer(NULL);
                    } break;
                case eDRAW_BOUNDING_DEBUG : 
                    {
                        DrawSphere(GetTransformation(), m_aabb);
                    } break;
                }
            }
            else
            {
                m_geo = std::static_pointer_cast<d3d::Geometry>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_ressource));
            }
        }
        else
        {
            VOnRestore(graph);
        }

    }

    UINT InstancedMeshNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO_INSTANCED | eDRAW_TO_SHADOW_MAP_INSTANCED | eDRAW_BOUNDING_DEBUG;
    }

    BOOL InstancedMeshNode::VIsVisible(SceneGraph* graph)
    {
        if(m_pInstanceHandle->IsReady())
        {
            BOOL in = graph->GetFrustum()->IsInside(m_transformedBBPoint, m_aabb.GetRadius());
            return in;
        }
        else
        {
            VOnRestore(graph);
        }
        return FALSE;
    }

    VOID InstancedMeshNode::VOnActorMoved(VOID)
    {
        m_transformedBBPoint = util::Mat4::Transform(*GetTransformation(), m_aabb.GetMiddle());
    }

    VOID InstancedMeshNode::VOnRestore(tbd::SceneGraph* graph)
    {
        MeshNode::VOnRestore(graph);

        std::shared_ptr<tbd::RenderComponent> comp = m_actor->GetComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID).lock();
        if(!comp->m_instances.empty())
        {
            FLOAT* instances = new FLOAT[comp->m_instances.size() * 3];
            INT index = 0;
            util::AxisAlignedBB& box = m_mesh->GetAABB();
            m_aabb = util::AxisAlignedBB();
            TBD_FOR(comp->m_instances)
            {
                instances[index++] = it->x;
                instances[index++] = it->y;
                instances[index++] = it->z;
                m_aabb.AddPoint(box.GetMin() + *it);
                m_aabb.AddPoint(box.GetMax() + *it);
            }
            m_aabb.Construct();

            m_pInstanceHandle = std::shared_ptr<d3d::VertexBufferHandle>(new d3d::VertexBufferHandle());
            m_pInstanceHandle->SetVertexData(instances, (UINT)comp->m_instances.size(), 3 * sizeof(FLOAT));
            std::string name("instanced_" + m_actorId);
            app::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(name, m_pInstanceHandle);
            SAFE_DELETE(instances);
        }
        else
        {
            LOG_CRITICAL_ERROR("this should not happen!");
        }
    }

    InstancedMeshNode::~InstancedMeshNode(VOID)
    {

    } 
}

