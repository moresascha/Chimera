#include "SceneNode.h"
#include "Event.h"
#include "Components.h"
#include "Quat.h"

namespace chimera 
{
    struct uint4
    {
        uint x, y, z, w;
    };

    void SetActorId(const IActor* id)
    {
        
    }

    void SetActorId(ActorId id)
    {

    }

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

    SceneNode::SceneNode(ActorId actorId) : m_actorId(actorId), m_wasVisibleOnLastTraverse(false), m_forceVisibleCheck(false), m_parent(NULL), m_paths(0)
    {
        VSetActor(actorId);
        m_wParentTransformation = std::unique_ptr<TransformComponent>(new TransformComponent());
    }

    SceneNode::SceneNode(void) : m_parent(NULL)
    {

    }

    ISceneNode* SceneNode::VGetParent(void)
    {
        if(HasParent())
        {
            return m_parent;
        }
        return NULL;
    }

    void SceneNode::VQueryGeometry(IGeometry** geo)
    {
        *geo = m_pGeometry.get();
    }

    void SceneNode::VSetActor(ActorId id)
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

    void SceneNode::VAddChild(std::unique_ptr<ISceneNode> child)
    {
        child->VSetParent(this);
        child->VOnParentChanged();
        m_childs.push_back(std::move(child));
    }

    uint SceneNode::VGetRenderPaths(void)
    {
        uint paths = 0;
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            paths |= (*it)->VGetRenderPaths();
        }
        return paths | m_paths;
    }

    void SceneNode::VSetRenderPaths(RenderPath paths)
    {
        m_paths = paths;
    }

    void SceneNode::VOnRestore(ISceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnRestore(graph);
        }
    }

    void SceneNode::VPostRender(ISceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPostRender(graph);
        }
    }

    void SceneNode::VPreRender(ISceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPreRender(graph);
        }
    }

    void SceneNode::VOnUpdate(ulong millis, ISceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnUpdate(millis, graph);
        }
    }

    void SceneNode::VSetParent(ISceneNode* parent)
    {
        m_parent = parent;
    }

    void SceneNode::VOnActorMoved(void)
    {
        TBD_FOR(m_childs)
        {
            VOnParentChanged();
        }
    }

    bool SceneNode::VIsVisible(ISceneGraph* graph) 
    {
        return true;
    }

    void SceneNode::VForceVisibilityCheck(void)
    {
        m_forceVisibleCheck = true;
    }

    void SceneNode::VRender(ISceneGraph* graph, RenderPath& path) 
    {
        if(graph->VIsVisibilityReset() || m_forceVisibleCheck)
        {
            VSetVisibilityOnLastTraverse(VIsVisible(graph));
            m_forceVisibleCheck = false;
        }

        if(VWasVisibleOnLastTraverse())
        {
            for(auto childIt = m_childs.begin(); childIt != m_childs.end(); ++childIt)
            {
                (*childIt)->VRender(graph, path);
            }
            VPreRender(graph);
            _VRender(graph, path);
            VPostRender(graph);
        }
    }

    bool SceneNode::VWasVisibleOnLastTraverse(void)
    {
        return m_wasVisibleOnLastTraverse;
    }

    bool SceneNode::HasParent(void)
    {
        return m_parent != NULL && m_parent->VGetActorId() != CM_INVALID_ACTOR_ID;
    }

    std::vector<std::unique_ptr<ISceneNode>>& SceneNode::VGetChilds(void)
    {
        return m_childs;
    }

    void SceneNode::VSetVisibilityOnLastTraverse(bool visible)
    {
        m_wasVisibleOnLastTraverse = visible;
    }

    ActorId SceneNode::VGetActorId(void)
    {
        return m_actorId;
    }

    std::unique_ptr<ISceneNode> SceneNode::VRemoveChild(ActorId actorId) 
    {
        if(m_actorId == actorId)
        {
            /*for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
            {
                (*it)->VSetParent(m_parent);
                m_parent->VAddChild(std::move(*it));
            }*/

            //todo: use map?
            for(auto i = m_parent->VGetChilds().begin(); i != m_parent->VGetChilds().end(); ++i)
            {
                if(i->get() == this)
                {
                    i->release();
                    m_parent->VGetChilds().erase(i);
                    break;
                }
            }
            return std::unique_ptr<ISceneNode>(this);
        }

        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            std::unique_ptr<ISceneNode> n = (*it)->VRemoveChild(actorId);
            if(n)
            {
                return n;
            }
        }
        return NULL;
    }

    std::unique_ptr<ISceneNode> SceneNode::VRemoveChild(ISceneNode* child) 
    {
        return std::move(VRemoveChild(child->VGetActorId()));
    }

    void SceneNode::ActorMovedDelegate(chimera::IEventPtr pEventData)
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

    util::Mat4* SceneNode::VGetTransformation(void)
    {
        if(HasParent())
        {
            return m_wParentTransformation->GetTransformation();
        }
        return m_transformation->GetTransformation();
    }

    void SceneNode::VOnParentChanged(void)
    {
        if(HasParent())
        {
            util::Mat4* pt = m_parent->VGetTransformation();
            util::Mat4* t = m_transformation->GetTransformation();
            
            util::Mat4* pwt = m_wParentTransformation->GetTransformation();

            *pwt = *pt;

            pwt->SetRotateQuat(pt->GetRotation());
            pwt->RotateQuat(t->GetRotation());

            util::Quat q(pt->GetRotation());
            util::Vec3 v = t->GetTranslation();

            q.Transform(v);

            pwt->SetTranslation(pt->GetTranslation());

            pwt->Translate(v);

            pwt->SetScale(t->GetScale() * pt->GetScale());

            VOnActorMoved();
        }

        TBD_FOR(m_childs)
        {
            (*it)->VOnParentChanged();
        }
    }

    SceneNode::~SceneNode(void)
    {
        REMOVE_EVENT_LISTENER(this, &SceneNode::ActorMovedDelegate, CM_EVENT_ACTOR_MOVED);
    }
}