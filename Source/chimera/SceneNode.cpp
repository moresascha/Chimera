#include "SceneNode.h"
#include "GameApp.h"
#include "Texture.h"
#include "VRamManager.h"
#include "GeometryFactory.h"
#include "D3DRenderer.h"
#include "Mat4.h"
#include "SceneGraph.h"
#include "D3DRenderer.h"
#include "GameView.h"
#include "GameLogic.h"
#include "tbdFont.h"
#include "EventManager.h"
#include "Frustum.h"
#include "Camera.h"
#include "Components.h"

namespace tbd 
{

    class tbd::SceneGraph;

    struct uint4
    {
        UINT x, y, z, w;
    };

    VOID DrawPickingSphere(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, FLOAT radius)
    {
        if(actor->HasComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID))
        {
            app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
            app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            GeometryFactory::GetSphere()->Bind();
            GeometryFactory::GetSphere()->Draw();
        }
    }

    VOID DrawPickingCube(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb)
    {
        if(actor->HasComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID))
        {
            util::Vec3 v = aabb.GetMax() - aabb.GetMin();
            v.Scale(0.5f);
            util::Mat4 m(*matrix);
            m.SetScale(v.x, v.y, v.z);
            m.Translate(aabb.GetMiddle().x, aabb.GetMiddle().y, aabb.GetMiddle().z);

            app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);
            app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
            GeometryFactory::GetGlobalBoundingBoxCube()->Bind();
            GeometryFactory::GetGlobalBoundingBoxCube()->Draw();
        }
    }

    VOID DrawSphere(CONST util::Mat4* matrix, CONST FLOAT radius)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eBoundingGeoBuffer);
        XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
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
        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*matrix);
        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eBoundingGeoBuffer);
        XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
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
        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        util::Mat4 m(*matrix);
        util::Vec3 v = aabb.GetMax() - aabb.GetMin();
        v.Scale(0.5f);
        m.SetScale(v.x, v.y, v.z);
        m.Translate(aabb.GetMiddle().x, aabb.GetMiddle().y, aabb.GetMiddle().z);

        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);

        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eBoundingGeoBuffer);
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

    VOID DrawAnchorSphere(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, FLOAT radius)
    {
        app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
        util::Mat4 m = *matrix;
        m.Scale(radius);
        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(m);
        app::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();
        app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(actor->GetId());
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
        app::g_pApp->GetFontManager()->RenderText(text.c_str(), pos.x, pos.y);
    }

    VOID DrawFrustum(tbd::Frustum& frustum)
    {
        d3d::Geometry* geo = GeometryFactory::GetFrustumGeometry();
        app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);

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

        app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(util::Mat4::IDENTITY);
        app::g_pApp->GetHumanView()->GetRenderer()->SetDefaultMaterial();

        geo->Bind();
        geo->Draw();

        app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
    }

    SceneNode::SceneNode(ActorId actorId) : m_actorId(actorId), m_wasVisibleOnLastTraverse(FALSE), m_forceVisibleCheck(FALSE)
    {
        VSetActor(actorId);
    }

    SceneNode::SceneNode(VOID) 
    {

    }

    VOID SceneNode::VSetActor(ActorId id)
    {
        if(id != INVALID_ACTOR_ID)
        {
            m_actorId = id;
            this->m_actor = app::g_pApp->GetLogic()->VFindActor(m_actorId);
            this->m_transformation = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
            if(!m_transformation)
            {
                LOG_CRITICAL_ERROR("Actor has no Transformcomponent");
            }
            event::EventListener listener = fastdelegate::MakeDelegate(this, &SceneNode::ActorMovedDelegate);
            app::g_pApp->GetEventMgr()->VAddEventListener(listener, event::ActorMovedEvent::TYPE);
        }
    }

    VOID SceneNode::VAddChild(std::shared_ptr<ISceneNode> child)
    {
        this->m_childs.push_back(child);
        std::shared_ptr<SceneNode> kid = std::static_pointer_cast<SceneNode>(child);
        kid->m_parent = this;
    }

    UINT SceneNode::VGetRenderPaths(VOID)
    {
        UINT paths = 0;
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            paths |= (*it)->VGetRenderPaths();
        }
        return paths;
    }

    VOID SceneNode::VOnRestore(tbd::SceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnRestore(graph);
        }
    }

    VOID SceneNode::VPostRender(tbd::SceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPostRender(graph);
        }
    }

    VOID SceneNode::VPreRender(tbd::SceneGraph* graph) 
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VPreRender(graph);
        }
    }

    VOID SceneNode::VOnUpdate(ULONG millis, SceneGraph* graph)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VOnUpdate(millis, graph);
        }
    }

    BOOL SceneNode::VRemoveChild(ActorId actorId) 
    {
        if(this->m_actorId == actorId)
        {
            return TRUE;
        }
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            if((*it)->VRemoveChild(actorId))
            {
                m_childs.erase(it);
                return TRUE;
            }
        }
        return FALSE;
    }

    BOOL SceneNode::VIsVisible(SceneGraph* graph) 
    {
        return TRUE; // TODO
    }

    VOID SceneNode::VForceVisibilityCheck(VOID)
    {
        m_forceVisibleCheck = TRUE;
    }

    VOID SceneNode::VRender(tbd::SceneGraph* graph, tbd::RenderPath& path) 
    {
        if(graph->IsVisibilityReset() || m_forceVisibleCheck)
        {
            VSetVisibilityOnLastTraverse(VIsVisible(graph));
            m_forceVisibleCheck = FALSE;
        }

        /*if(m_actorId == 3)
        {
            DEBUG_OUT_A("%d, %d", graph->IsVisibilityReset(), VWasVisibleOnLastTraverse());
        } */

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

    VOID SceneNode::VSetVisibilityOnLastTraverse(BOOL visible)
    {
        m_wasVisibleOnLastTraverse = visible;
    }

    ActorId SceneNode::VGetActorId(VOID)
    {
        return m_actorId;
    }

    BOOL SceneNode::VRemoveChild(std::shared_ptr<ISceneNode> child) 
    {
        return this->VRemoveChild(child->VGetActorId());
    }

    VOID SceneNode::VRenderChildren(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        for(auto it = m_childs.begin(); it != m_childs.end(); ++it)
        {
            (*it)->VRender(graph, path);
        } 
    }

    VOID SceneNode::ActorMovedDelegate(event::IEventPtr pEventData)
    {
        std::shared_ptr<event::ActorMovedEvent> movedEvent = std::static_pointer_cast<event::ActorMovedEvent>(pEventData);
        if(movedEvent->m_actor->GetId() == m_actor->GetId())
        {
            VForceVisibilityCheck();
            VOnActorMoved();
        }
    }

    SceneNode::~SceneNode(VOID)
    {
        event::EventListener listener = fastdelegate::MakeDelegate(this, &SceneNode::ActorMovedDelegate);
        app::g_pApp->GetEventMgr()->VRemoveEventListener(listener, event::ActorMovedEvent::TYPE);
    }
}