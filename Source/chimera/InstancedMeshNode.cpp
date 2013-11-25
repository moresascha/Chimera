#include "SceneNode.h"
#include "Components.h"
#include "Frustum.h"

namespace chimera
{

//     class VertexBufferHandle : public VRamHandle
//     {
// 
//     private:
//         IVertexBuffer* m_pVertexBuffer;
// 
//     public:
// 
//         VertexBufferHandle(std::shared_ptr<IVertexBuffer> instances) : m_pVertexBuffer(NULL)
//         {
//         }
// 
//         UINT VertexBufferHandle::VGetByteCount(VOID) CONST
//         {
//             return m_pVertexBuffer->VGetByteCount() * sizeof(FLOAT);
//         }
// 
//         VOID VertexBufferHandle::SetVertexData(CONST VOID* data, UINT vertexCount, UINT stride, BOOL cpuAccessFlags /* = */ )
//         {
//             if(!m_pVertexBuffer)
//             {
//                 m_pVertexBuffer = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer().get();
//                 m_pVertexBuffer->VCreate(vertexCount, stride, data, cpuAccessFlags);
//             }
//             m_pVertexBuffer->VSetData(data, vertexCount);
//         }
// 
//         BOOL VertexBufferHandle::VCreate(VOID)
//         {
//             m_pVertexBuffer->VCreate();
//             return TRUE;
//         }
// 
//         VOID VertexBufferHandle::VDestroy(VOID)
//         {
//             SAFE_DELETE(m_pVertexBuffer);
//         }
// 
//         IDeviceBuffer* VertexBufferHandle::GetBuffer(VOID) CONST
//         {
//             return m_pVertexBuffer;
//         }
//     };

    InstancedMeshNode::InstancedMeshNode(ActorId actorid, std::shared_ptr<IVertexBuffer> instances, CMResource ressource) : MeshNode(actorid, ressource), m_pInstances(instances)
    {
        RenderPath p = VGetRenderPaths();
        p ^= CM_RENDERPATH_ALBEDO;
        p ^= CM_RENDERPATH_SHADOWMAP;
        p |= CM_RENDERPATH_ALBEDO_INSTANCED;
        p |= CM_RENDERPATH_SHADOWMAP_INSTANCED;
        VSetRenderPaths(p);
    }

    VOID InstancedMeshNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        switch(path)
        {
        case CM_RENDERPATH_ALBEDO_INSTANCED :
            {
                RenderPath rp = CM_RENDERPATH_ALBEDO;
                m_pGeometry->VSetInstanceBuffer(m_pInstances.get());
                MeshNode::_VRender(graph, rp);
                m_pGeometry->VSetInstanceBuffer(NULL);
            } break;

        case CM_RENDERPATH_SHADOWMAP_INSTANCED :
            {
                RenderPath rp = CM_RENDERPATH_SHADOWMAP;
                m_pGeometry->VSetInstanceBuffer(m_pInstances.get());
                MeshNode::_VRender(graph, rp);
                m_pGeometry->VSetInstanceBuffer(NULL);
            } break;
        }
    }

    BOOL InstancedMeshNode::VIsVisible(ISceneGraph* graph)
    {
        BOOL in = graph->VGetFrustum()->IsInside(m_transformedBBPoint, m_aabb.GetRadius());
        return TRUE;
    }

    VOID InstancedMeshNode::VOnActorMoved(VOID)
    {
        m_transformedBBPoint = util::Mat4::Transform(*VGetTransformation(), m_aabb.GetMiddle());
    }

    VOID InstancedMeshNode::VOnRestore(ISceneGraph* graph)
    {
        MeshNode::VOnRestore(graph);

//         RenderComponent* comp = GetActorCompnent<RenderComponent>(m_actor, CM_CMP_RENDERING);
//         if(!comp->m_instances.empty())
//         {
//             FLOAT* instances = new FLOAT[comp->m_instances.size() * 3];
//             INT index = 0;
//             util::AxisAlignedBB& box = m_mesh->VGetAABB();
//             m_aabb = util::AxisAlignedBB();
//             TBD_FOR(comp->m_instances)
//             {
//                 instances[index++] = it->x;
//                 instances[index++] = it->y;
//                 instances[index++] = it->z;
//                 m_aabb.AddPoint(box.GetMin() + *it);
//                 m_aabb.AddPoint(box.GetMax() + *it);
//             }
//             m_aabb.Construct();
// 
//             m_pInstanceHandle = std::shared_ptr<IVertexBufferHandle>(new chimera::VertexBufferHandle());
//             m_pInstanceHandle->SetVertexData(instances, (UINT)comp->m_instances.size(), 3 * sizeof(FLOAT));
//             std::string name("instanced_" + m_actorId);
//             chimera::g_pApp->GetHumanView()->GetVRamManager()->AppendAndCreateHandle(name, m_pInstanceHandle);
//             SAFE_DELETE(instances);
//         }
    }

    InstancedMeshNode::~InstancedMeshNode(VOID)
    {

    } 
}

