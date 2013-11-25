#include "SceneNode.h"
#include "Event.h"
#include "Components.h"

namespace chimera 
{
    struct uint4
    {
        UINT x, y, z, w;
    };

    /*

    VOID DrawPickingSphere(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, FLOAT radius)
    {
        if(actor->HasComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID))
        {
            chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
            chimera::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            GeometryFactory::GetSphere()->Bind();
            GeometryFactory::GetSphere()->Draw();
        }
    }

    VOID DrawPickingCube(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb)
    {
        if(actor->HasComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID))
        {
            util::Vec3 v = aabb.GetMax() - aabb.GetMin();
            v.Scale(0.5f);
            util::Mat4 m(*matrix);
            m.SetScale(v.x, v.y, v.z);
            m.Translate(aabb.GetMiddle().x, aabb.GetMiddle().y, aabb.GetMiddle().z);

            chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);
            chimera::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            GeometryFactory::GetGlobalBoundingBoxCube()->Bind();
            GeometryFactory::GetGlobalBoundingBoxCube()->Draw();
        }
    }

    VOID DrawSphere(CONST util::Mat4* matrix, CONST FLOAT radius)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
        XMFLOAT4* f = (XMFLOAT4*)buffer->Map();
        f->x = radius;
        f->y = 0;
        f->z = 0;
        f->w = 0;
        buffer->Unmap();
        GeometryFactory::GetSphere()->Bind();
        GeometryFactory::GetSphere()->Draw();
    }

    VOID DrawSphere(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb)
    {
        if(aabb.GetRadius() > 900) //todo
        {
            return;
        }
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
        XMFLOAT4* f = (XMFLOAT4*)buffer->Map();
        f->x = aabb.GetRadius();
        f->y = aabb.GetMiddle().x;
        f->z = aabb.GetMiddle().y;
        f->w = aabb.GetMiddle().z;
        buffer->Unmap();
        GeometryFactory::GetSphere()->Bind();
        GeometryFactory::GetSphere()->Draw();
    }

    VOID DrawBox(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        
        util::Mat4 m(*matrix);
        util::Vec3 v = aabb.GetMax() - aabb.GetMin();
        v.Scale(0.5f);
        m.SetScale(v.x, v.y, v.z);
        m.Translate(aabb.GetMiddle().x, aabb.GetMiddle().y, aabb.GetMiddle().z);

        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);

        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
        //aabb.GetMiddle().Print();
        XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
        f->x = 1;//aabb.GetRadius();
        f->y = 0;//aabb.GetMiddle().x;
        f->z = 0;//aabb.GetMiddle().y;
        f->w = 0;//aabb.GetMiddle().z;
        buffer->Unmap();
        GeometryFactory::GetGlobalBoundingBoxCube()->Bind();
        GeometryFactory::GetGlobalBoundingBoxCube()->Draw();
    }

    VOID DrawBox(CONST util::AxisAlignedBB& aabb)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        util::Vec3 v = aabb.GetMax() - aabb.GetMin();
        v.Scale(0.5f);
        util::Mat4 m;
        m.SetScale(v.x, v.y, v.z);
        m.Translate(aabb.GetMiddle().x, aabb.GetMiddle().y, aabb.GetMiddle().z);

        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);

        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
        //aabb.GetMiddle().Print();
        XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
        f->x = 1;//aabb.GetRadius();
        f->y = 0;//aabb.GetMiddle().x;
        f->z = 0;//aabb.GetMiddle().y;
        f->w = 0;//aabb.GetMiddle().z;
        buffer->Unmap();
        GeometryFactory::GetGlobalBoundingBoxCube()->Bind();
        GeometryFactory::GetGlobalBoundingBoxCube()->Draw();
    }

    VOID DrawAnchorSphere(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, FLOAT radius)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        util::Mat4 m = *matrix;
        m.Scale(radius);
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
        GeometryFactory::GetSphere()->Bind();
        GeometryFactory::GetSphere()->Draw();
    }

    VOID DrawInfoTextOnScreen(util::ICamera* camera, CONST util::Mat4* model, CONST std::string& text)
    {
        util::Vec3 pos(model->GetTranslation());
        pos = util::Mat4::Transform(camera->GetView(), pos);
        pos = util::Mat4::Transform(camera->GetProjection(), pos);
        if(pos.z < -1 || pos.z > 1)
        {
            return;
        }
        pos.x = 0.5f * pos.x + 0.5f;
        pos.y = 0.5f * (-pos.y) + 0.5f;
        chimera::g_pApp->GetFontManager()->RenderText(text.c_str(), pos.x, pos.y);
    }

    VOID DrawFrustum(chimera::Frustum& frustum)
    {
        chimera::Geometry* geo = GeometryFactory::GetFrustumGeometry();
        chimera::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(chimera::g_pRasterizerStateWrireframe);

        FLOAT vertexTmp[8 * 24] = 
        {
            //vorne
            frustum.GetPoints()[leftDownNear].x, frustum.GetPoints()[leftDownNear].y, frustum.GetPoints()[leftDownNear].z, 0, 0, -1,  0, 1,
            frustum.GetPoints()[leftUpNear].x, frustum.GetPoints()[leftUpNear].y, frustum.GetPoints()[leftUpNear].z, 0, 0, -1,  0, 0,
            frustum.GetPoints()[rightUpNear].x, frustum.GetPoints()[rightUpNear].y, frustum.GetPoints()[rightUpNear].z, 0, 0, -1,  1, 0,
            frustum.GetPoints()[rightDownNear].x, frustum.GetPoints()[rightDownNear].y, frustum.GetPoints()[rightDownNear].z, 0, 0, -1,  1, 1,
            
            //hinten
            frustum.GetPoints()[leftDownFar].x, frustum.GetPoints()[leftDownFar].y, frustum.GetPoints()[leftDownFar].z, 0, 0, +1,  1, 0,
            frustum.GetPoints()[leftUpFar].x, frustum.GetPoints()[leftUpFar].y, frustum.GetPoints()[leftUpFar].z, 0, 0, +1,  1, 1,
            frustum.GetPoints()[rightUpFar].x, frustum.GetPoints()[rightUpFar].y, frustum.GetPoints()[rightUpFar].z, 0, 0, +1,  0, 1,
            frustum.GetPoints()[rightDownFar].x, frustum.GetPoints()[rightDownFar].y, frustum.GetPoints()[rightDownFar].z, 0, 0, +1,  0, 0,
            
            //links
            frustum.GetPoints()[leftDownNear].x, frustum.GetPoints()[leftDownNear].y, frustum.GetPoints()[leftDownNear].z, -1, 0, 0,  1, 0, //8
            frustum.GetPoints()[leftDownFar].x, frustum.GetPoints()[leftDownFar].y, frustum.GetPoints()[leftDownFar].z, -1, 0, 0,  0, 0, //9
            frustum.GetPoints()[leftUpNear].x, frustum.GetPoints()[leftUpNear].y, frustum.GetPoints()[leftUpNear].z, -1, 0, 0,  1, 1, //10
            frustum.GetPoints()[leftUpFar].x, frustum.GetPoints()[leftUpFar].y, frustum.GetPoints()[leftUpFar].z, -1, 0, 0,  0, 1, //11
          
            //rechts
            frustum.GetPoints()[rightDownNear].x, frustum.GetPoints()[rightDownNear].y, frustum.GetPoints()[rightDownNear].z, +1, 0, 0,  0, 0, //12
            frustum.GetPoints()[rightDownFar].x, frustum.GetPoints()[rightDownFar].y, frustum.GetPoints()[rightDownFar].z,+1, 0, 0,  1, 0, //13
            frustum.GetPoints()[rightUpNear].x, frustum.GetPoints()[rightUpNear].y, frustum.GetPoints()[rightUpNear].z, +1, 0, 0,  0, 1, //14
            frustum.GetPoints()[rightUpFar].x, frustum.GetPoints()[rightUpFar].y, frustum.GetPoints()[rightUpFar].z, +1, 0, 0,  1, 1, //15
              
            //oben
            frustum.GetPoints()[rightUpFar].x, frustum.GetPoints()[rightUpFar].y, frustum.GetPoints()[rightUpFar].z, 0, +1, 0,  1, 1, //16
            frustum.GetPoints()[rightUpNear].x, frustum.GetPoints()[rightUpNear].y, frustum.GetPoints()[rightUpNear].z, 0, +1, 0,  1, 0, //17
            frustum.GetPoints()[leftUpFar].x, frustum.GetPoints()[leftUpFar].y, frustum.GetPoints()[leftUpFar].z, 0, +1, 0,  0, 1, //18
            frustum.GetPoints()[leftUpNear].x, frustum.GetPoints()[leftUpNear].y, frustum.GetPoints()[leftUpNear].z, 0, +1, 0,  0, 0, //19
            
            //unten
            frustum.GetPoints()[rightDownFar].x, frustum.GetPoints()[rightDownFar].y, frustum.GetPoints()[rightDownFar].z, 0, -1, 0,  0, 1, //20
            frustum.GetPoints()[rightDownNear].x, frustum.GetPoints()[rightDownNear].y, frustum.GetPoints()[rightDownNear].z, 0, -1, 0,  0, 0, //21
            frustum.GetPoints()[leftDownFar].x, frustum.GetPoints()[leftDownFar].y, frustum.GetPoints()[leftDownFar].z, 0, -1, 0,  1, 1, //22
            frustum.GetPoints()[leftDownNear].x, frustum.GetPoints()[leftDownNear].y, frustum.GetPoints()[leftDownNear].z, 0, -1, 0,  1, 0, //23

        };

        D3D11_MAPPED_SUBRESOURCE* ress = geo->GetVertexBuffer()->Map();
        memcpy(ress->pData, vertexTmp, 24 * 8 * sizeof(FLOAT));
        geo->GetVertexBuffer()->Unmap();

        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(util::Mat4::IDENTITY);
        chimera::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();

        geo->Bind();
        geo->Draw();

        chimera::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    } */

    SceneNode::SceneNode(ActorId actorId) : m_actorId(actorId), m_wasVisibleOnLastTraverse(FALSE), m_forceVisibleCheck(FALSE), m_parent(NULL), m_paths(0)
    {
        VSetActor(actorId);
        m_wParentTransformation = std::unique_ptr<TransformComponent>(new TransformComponent());
    }

    SceneNode::SceneNode(VOID) : m_parent(NULL)
    {

    }

    VOID SceneNode::VQueryGeometry(IGeometry** geo)
    {
        *geo = m_pGeometry.get();
    }

    VOID SceneNode::VSetActor(ActorId id)
    {
        if(id != CM_INVALID_ACTOR_ID)
        {
            m_actorId = id;
            m_actor = CmGetApp()->VGetLogic()->VFindActor(m_actorId);
            m_transformation = GetActorCompnent<TransformComponent>(m_actor, CM_CMP_TRANSFORM);
            if(!m_transformation)
            {
                LOG_CRITICAL_ERROR("Actor has no Transformcomponent");
            }
            ADD_EVENT_LISTENER(this, &SceneNode::ActorMovedDelegate, CM_EVENT_ACTOR_MOVED);
        }
    }

    VOID SceneNode::VAddChild(std::unique_ptr<ISceneNode> child)
    {
        child->VSetParent(this);
        child->VOnParentChanged();
        m_childs.push_back(std::move(child));
    }

    UINT SceneNode::VGetRenderPaths(VOID)
    {
        UINT paths = 0;
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            paths |= (*it)->VGetRenderPaths();
        }
        return paths | m_paths;
    }

    VOID SceneNode::VSetRenderPaths(RenderPath paths)
    {
        m_paths = paths;
    }

    VOID SceneNode::VOnRestore(ISceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnRestore(graph);
        }
    }

    VOID SceneNode::VPostRender(ISceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPostRender(graph);
        }
    }

    VOID SceneNode::VPreRender(ISceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPreRender(graph);
        }
    }

    VOID SceneNode::VOnUpdate(ULONG millis, ISceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnUpdate(millis, graph);
        }
    }

    VOID SceneNode::VSetParent(ISceneNode* parent)
    {
        m_parent = parent;
    }

    BOOL SceneNode::VRemoveChild(ActorId actorId) 
    {
        if(m_actorId == actorId)
        {
            for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
            {
                (*it)->VSetParent(m_parent);
                m_parent->VAddChild(std::move(*it));
            }
            return TRUE;
        }
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            if((*it)->VRemoveChild(actorId))
            {
                return TRUE;
            }
        }
        return FALSE;
    }

    BOOL SceneNode::VIsVisible(ISceneGraph* graph) 
    {
        return TRUE;
    }

    VOID SceneNode::VForceVisibilityCheck(VOID)
    {
        m_forceVisibleCheck = TRUE;
    }

    VOID SceneNode::VRender(ISceneGraph* graph, RenderPath& path) 
    {
        if(graph->VIsVisibilityReset() || m_forceVisibleCheck)
        {
            VSetVisibilityOnLastTraverse(VIsVisible(graph));
            m_forceVisibleCheck = FALSE;
        }

        if(VWasVisibleOnLastTraverse())
        {
            VPreRender(graph);
            _VRender(graph, path);
            VPostRender(graph);

            VRenderChildren(graph, path);
        }
    }

    BOOL SceneNode::VWasVisibleOnLastTraverse(VOID)
    {
        return m_wasVisibleOnLastTraverse;
    }

    BOOL SceneNode::HasParent(VOID)
    {
        return m_parent != NULL && m_parent->VGetActorId() != CM_INVALID_ACTOR_ID;
    }

    std::vector<std::unique_ptr<ISceneNode>>& SceneNode::VGetChilds(VOID)
    {
        return m_childs;
    }

    VOID SceneNode::VSetVisibilityOnLastTraverse(BOOL visible)
    {
        m_wasVisibleOnLastTraverse = visible;
    }

    ActorId SceneNode::VGetActorId(VOID)
    {
        return m_actorId;
    }

    BOOL SceneNode::VRemoveChild(ISceneNode* child) 
    {
        return VRemoveChild(child->VGetActorId());
    }

    VOID SceneNode::VRenderChildren(ISceneGraph* graph, RenderPath& path)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VRender(graph, path);
        } 
    }

    VOID SceneNode::ActorMovedDelegate(chimera::IEventPtr pEventData)
    {
        std::shared_ptr<ActorMovedEvent> movedEvent = std::static_pointer_cast<ActorMovedEvent>(pEventData);
        if(movedEvent->m_actor->GetId() == m_actor->GetId())
        {
            VForceVisibilityCheck();
            VOnActorMoved();

            TBD_FOR(m_childs)
            {
                (*it)->VOnParentChanged();
            }
        }
    }

    util::Mat4* SceneNode::VGetTransformation(VOID)
    {
        if(HasParent())
        {
            return m_wParentTransformation->GetTransformation();
        }
        return m_transformation->GetTransformation();
    }

    VOID SceneNode::VOnParentChanged(VOID)
    {
        if(HasParent())
        {
            util::Mat4* pt = m_parent->VGetTransformation();
            util::Mat4* t = m_transformation->GetTransformation();
            
            util::Mat4* pwt = m_wParentTransformation->GetTransformation();

            *pwt = *pt;

            pwt->RotateQuat(t->GetRotation());

            pwt->Translate(t->GetTranslation());

            pwt->Scale(t->GetScale());

            VOnActorMoved();
        }
        TBD_FOR(m_childs)
        {
            VOnParentChanged();
        }
    }

    ISceneNode* SceneNode::VFindActor(ActorId id)
    {
        TBD_FOR(m_childs)
        {
            ISceneNode* node = it->get();
            if(node->VGetActorId() == id)
            {
                return node;
            }
        }
        TBD_FOR(m_childs)
        {
            ISceneNode* node = it->get();
            ISceneNode* pNode = node->VFindActor(id);
            if(pNode)
            {
                pNode;
            }
        }
        return NULL;
    }

    SceneNode::~SceneNode(VOID)
    {
        REMOVE_EVENT_LISTENER(this, &SceneNode::ActorMovedDelegate, CM_EVENT_ACTOR_MOVED);
    }
}