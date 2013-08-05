#include "PhysicsSystem.h"
#include "Components.h"
#include "GameApp.h"
#include "physx/PxToolkit.h"
#include "GameLogic.h"
#include "EventManager.h"
#include "Mesh.h"
#include "math.h"

#include "GameApp.h"
#include "GameView.h"
#include "SceneGraph.h"
#include "Camera.h"

#define NDEBUG

#ifdef _DEBUG
    #pragma comment (lib, "PhysX3CHECKED_x64.lib")
    #pragma comment (lib, "PhysX3CommonCHECKED_x64.lib")
    #pragma comment (lib, "PhysX3CookingCHECKED_x64.lib")
    #pragma comment (lib, "PhysXProfileSDKCHECKED.lib")
    #pragma comment (lib, "PhysX3ExtensionsCHECKED.lib")
    #pragma comment (lib, "PxToolkitDEBUG.lib")
    #pragma comment (lib, "PhysX3CharacterKinematicCHECKED_x64.lib")
    #pragma comment (lib, "PhysXVisualDebuggerSDKCHECKED.lib")
#else 
    #pragma comment (lib, "PhysX3_x64.lib")
    #pragma comment (lib, "PhysX3Common_x64.lib")
    #pragma comment (lib, "PhysX3Cooking_x64.lib")
    #pragma comment (lib, "PhysX3Extensions.lib")
    #pragma comment (lib, "PhysXProfileSDK.lib")
    #pragma comment (lib, "PxToolkit.lib")
    #pragma comment (lib, "PhysX3CharacterKinematic_x64.lib")
    #pragma comment (lib, "PhysXVisualDebuggerSDK.lib")
#endif

namespace logic 
{

    using namespace physx;

    PxFilterFlags DefaultFilterShader(
        PxFilterObjectAttributes attributes0, PxFilterData filterData0,
        PxFilterObjectAttributes attributes1, PxFilterData filterData1,
        PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
    {
        // let triggers through
        if(PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1))
        {
            pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
            return PxFilterFlag::eDEFAULT;
        }
        // generate contacts for all that were not filtered above
        pairFlags = PxPairFlag::eCONTACT_DEFAULT;

        // trigger the contact callback for pairs (A,B) where
        // the filtermask of A contains the ID of B and vice versa.
        //if((filterData0.word0 & filterData1.word1) && (filterData1.word0 & filterData0.word1))
        {
            pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND;
        }

        return PxFilterFlag::eDEFAULT;
    }


    class DefaultFilterCallback : public physx::PxSimulationEventCallback
    {
    private:
        PhysX* m_pPhysix;

    public:
        DefaultFilterCallback(PhysX* physx) : m_pPhysix(physx)
        {

        }
        VOID onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)
        {

        }
        VOID onWake(PxActor** actors, PxU32 count)
        {

        }
        VOID onSleep(PxActor** actors, PxU32 count)
        {

        }
        VOID onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs)
        {
            for(PxU32 i = 0; i < nbPairs; ++i)
            {
                const PxContactPair& cp = pairs[i];

                if(cp.events & PxPairFlag::eNOTIFY_TOUCH_FOUND)
                {
                    ActorId id0 = m_pPhysix->m_pxActorToActorId[pairHeader.actors[0]];
                    ActorId id1 = m_pPhysix->m_pxActorToActorId[pairHeader.actors[1]];
                    event::CollisionEvent* ce = new event::CollisionEvent();
                    ce->m_actor0 = id0;
                    ce->m_actor1 = id1;
                    QUEUE_EVENT(ce);
                }
            }
        }

        VOID onTrigger(PxTriggerPair* pairs, PxU32 count)
        {
            for(PxU32 i = 0; i < count; i++)
            {
                if (pairs[i].flags & (PxTriggerPairFlag::eDELETED_SHAPE_TRIGGER | PxTriggerPairFlag::eDELETED_SHAPE_OTHER))
                {
                    continue;
                }

                physx::PxActor* a0 = &pairs[i].triggerShape->getActor();
                physx::PxActor* a1 = &pairs[i].otherShape->getActor();
                ActorId id0 = m_pPhysix->m_pxActorToActorId[a0];
                ActorId id1 = m_pPhysix->m_pxActorToActorId[a1];
                event::TriggerEvent* te = new event::TriggerEvent();
                te->m_triggerActor = id0;
                te->m_didTriggerActor = id1;
                QUEUE_EVENT(te);
            }
        }
    };


    PhysX::PhysX(VOID) : m_pCooking(NULL), m_pDesc(NULL), m_pFoundation(NULL), m_pPhysx(NULL), m_pProfileManager(NULL), m_pScene(NULL), m_pControllerManager(NULL), m_pDebugConnection(NULL)
    {
        m_pDefaultFilterCallback = new DefaultFilterCallback(this);
    }

    BOOL PhysX::VInit(VOID) 
    {
        m_pFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, m_allocator, m_errorCallback);
        if(!m_pFoundation)
        {
            return FALSE;
        }

        m_pProfileManager = &physx::PxProfileZoneManager::createProfileZoneManager(m_pFoundation);
        if(!m_pProfileManager)
        {
            return FALSE;
        }

        bool recordMemoryAllocations = FALSE;
    
    #ifdef _DEBUG
        recordMemoryAllocations = TRUE;
    #endif

        m_pPhysx = PxCreatePhysics(PX_PHYSICS_VERSION, *m_pFoundation, physx::PxTolerancesScale(), recordMemoryAllocations, m_pProfileManager);
        if(!m_pPhysx)
        {
            return FALSE;
        }

        m_pControllerManager = PxCreateControllerManager(*m_pFoundation);
        RETURN_IF_FAILED(m_pControllerManager);

        m_pCooking = PxCreateCooking(PX_PHYSICS_VERSION, *m_pFoundation, physx::PxCookingParams());
        if(!m_pCooking)
        {
            return FALSE;
        }

        m_pDesc = new physx::PxSceneDesc(this->m_pPhysx->getTolerancesScale());
        m_pDesc->gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);

        m_pDesc->filterShader = DefaultFilterShader;//physx::PxDefaultSimulationFilterShader;
  
        m_pDesc->cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(4);

        m_pDesc->flags |= physx::PxSceneFlag::eENABLE_ACTIVETRANSFORMS | physx::PxSceneFlag::eENABLE_KINEMATIC_PAIRS | physx::PxSceneFlag::eENABLE_KINEMATIC_STATIC_PAIRS;

        m_pScene = this->m_pPhysx->createScene(*m_pDesc);

        if(!m_pScene)
        {
            return FALSE;
        }

        m_pScene->setSimulationEventCallback(m_pDefaultFilterCallback);

        /*if(!PxInitExtensions(*m_pPhysx))
        {
            return FALSE;
        } */

        //set default materials
        m_materials["dynamic"] = Material(0.5f, 0.5f, 0.1f, 1.0f, 1.0f, m_pPhysx);
        m_materials["static"] = Material(0.5f, 0.5f, 0.1f, 0.0f, 1.0f, m_pPhysx);
        m_materials["bouncy"] = Material(0.5f, 0.5f, 0.8f, 1.0f, 2.0f, m_pPhysx);
        m_materials["kinematic"] = Material(0.5f, 0.5f, 0.8f, 1.0f, 2.0f, m_pPhysx);
        m_materials["default"] = Material(0.5f, 0.5f, 0.1f, 1.0f, 2.0f, m_pPhysx);
        //m_materials["bouncy"].m_material->setFlag(physx::PxMaterialFlag::eDISABLE_STRONG_FRICTION, FALSE);
        //end

        event::EventListener listener = fastdelegate::MakeDelegate(this, &PhysX::NewComponentDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::NewComponentCreatedEvent::TYPE);

        ADD_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, event::ApplyForceEvent::TYPE);
        ADD_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, event::ApplyTorqueEvent::TYPE);

    #ifdef _DEBUG
        // DEBUGGING
        if(m_pPhysx->getPvdConnectionManager() == NULL)
            return FALSE;

   
        // setup connection parameters
        const char*     pvd_host_ip = "127.0.0.1";  // IP of the PC which is running PVD
        int             port        = 5425;         // TCP port to connect to, where PVD is listening
        unsigned int    timeout     = 100;          // timeout in milliseconds to wait for PVD to respond,
        // consoles and remote PCs need a higher timeout.
        physx::PxVisualDebuggerConnectionFlags connectionFlags = physx::PxVisualDebuggerExt::getAllConnectionFlags();

        // and now try to connect
        m_pDebugConnection = physx::PxVisualDebuggerExt::createConnection(m_pPhysx->getPvdConnectionManager(),
            pvd_host_ip, port, timeout, connectionFlags);
    #endif
        // remember to release the connection by manual in the end


        return TRUE;
    }

    VOID PhysX::VCreateStaticPlane(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, std::string& material) {
    
        physx::PxMaterial* mat = m_materials[material].m_material;
        std::shared_ptr<tbd::TransformComponent> comp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
        util::Vec3 trans = comp->GetTransformation()->GetTranslation();
        physx::PxRigidStatic* plane = PxCreatePlane(*m_pPhysx, physx::PxPlane(physx::PxVec3(trans.x, trans.y, trans.z), physx::PxVec3(0, 1, 0)), *mat);

        m_pScene->addActor(*plane);

        this->m_actorIdToPxActorMap[actor->GetId()].push_back(plane);
        this->m_pxActorToActorId[plane] = actor->GetId();
    }

    VOID PhysX::VCreateSphere(FLOAT radius, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material) {
        physx::PxSphereGeometry geo(radius);
        this->AddActor(geo, actor, offsetPosition, material, physx::PxPi * 4.0f/3.0f * radius * radius * radius);
    }

    VOID PhysX::VCreateCube(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material) {
        physx::PxBoxGeometry geo(dimension.x * 0.5f, dimension.y * 0.5f, dimension.z * 0.5f);
        this->AddActor(geo, actor, offsetPosition, material, dimension.x * dimension.y * dimension.z);
    }

    VOID PhysX::VCreateTriangleMesh(std::shared_ptr<tbd::Actor> actor, CONST tbd::Mesh* mesh, CONST util::Vec3& offsetPosition, std::string& material)
    {
        physx::PxTriangleMeshDesc meshDesc;
        std::shared_ptr<tbd::RenderComponent> rc = actor->GetComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID).lock();
    
        UINT stride = mesh->GetVertexStride() / sizeof(UINT);
        UINT count = mesh->GetVertexCount();

        physx::PxVec3* verts = new physx::PxVec3[mesh->GetVertexCount()];

        for(UINT i = 0; i < mesh->GetVertexCount(); ++i)
        {
            physx::PxVec3 v;
            v.x = mesh->GetVertices()[i * stride + 0];
            v.y = mesh->GetVertices()[i * stride + 1];
            v.z = mesh->GetVertices()[i * stride + 2];
            verts[i] = v;
        }
        meshDesc.points.count = count;
        meshDesc.points.stride = sizeof(physx::PxVec3);
        meshDesc.points.data = verts;

        physx::PxU32* indices32 = new physx::PxU32[mesh->GetIndexCount()];
        UINT index = 0;
        for(UINT i = 0; i < mesh->GetIndexCount() / 3; ++i)
        {
            UINT i0 = mesh->GetIndices()[3 * i + 0];
            UINT i1 = mesh->GetIndices()[3 * i + 1];
            UINT i2 = mesh->GetIndices()[3 * i + 2];
            indices32[3 * i + 0] = i0;
            indices32[3 * i + 1] = i2;
            indices32[3 * i + 2] = i1;
            /*
            tbd::Face f = mesh->GetFaces()[i];
            if(f.m_triples.size() == 3)
            {
                indices32[index++] = f.m_triples[0].position;
                indices32[index++] = f.m_triples[1].position;
                indices32[index++] = f.m_triples[2].position;
            }
            else if(f.m_triples.size() == 4)
            {
                indices32[index++] = f.m_triples[0].position;
                indices32[index++] = f.m_triples[1].position;
                indices32[index++] = f.m_triples[2].position;

                indices32[index++] = f.m_triples[1].position;
                indices32[index++] = f.m_triples[2].position;
                indices32[index++] = f.m_triples[3].position;
            }
            else
            {
                LOG_ERROR("unkown triples count");
            } */
        }

        meshDesc.triangles.count = mesh->GetIndexCount() / 3;
        meshDesc.triangles.stride = 3 * sizeof(physx::PxU32);
        meshDesc.triangles.data = indices32;

        PxToolkit::MemoryOutputStream writeBuffer;
        if(!m_pCooking->cookTriangleMesh(meshDesc, writeBuffer))
        {
            LOG_CRITICAL_ERROR("failed to cook triangle mesh");
        }

        PxToolkit::MemoryInputData readBuffer(writeBuffer.getData(), writeBuffer.getSize());
        physx::PxTriangleMesh* pxmesh = m_pPhysx->createTriangleMesh(readBuffer);
        physx::PxMeshScale scale;
        scale.scale.z = scale.scale.y = scale.scale.x = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetScale().x;
        physx::PxTriangleMeshGeometry geo(pxmesh, scale);

        AddActor(geo, actor, offsetPosition, material, 1);

        SAFE_ARRAY_DELETE(indices32);
        SAFE_ARRAY_DELETE(verts);
    }

    VOID PhysX::VCreateTrigger(FLOAT radius, std::shared_ptr<tbd::Actor> actor) 
    {
        physx::PxSphereGeometry geo(radius);
        std::string m("static");
        physx::PxActor* a = AddActor(geo, actor, util::Vec3(0,0,0), m, physx::PxPi * 4.0f/3.0f * radius * radius * radius);
        physx::PxShape* shape;
        ((physx::PxRigidStatic*)a)->getShapes(&shape, 1);
        shape->setFlag(PxShapeFlag::eTRIGGER_SHAPE, TRUE);
    }

    VOID PhysX::VRemoveActor(ActorId id) 
    {
        auto it = m_actorIdToPxActorMap.find(id);
        if(it != m_actorIdToPxActorMap.end())
        {
            std::vector<physx::PxActor*>& list = it->second;
            for(auto itt = list.begin(); itt != list.end(); ++itt)
            {
                m_pxActorToActorId.erase(*itt);
                m_pScene->removeActor(**itt);
            }
            m_actorIdToPxActorMap.erase(id);
        }
    }

    VOID PhysX::VApplyForce(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor) 
    {
        auto iit = m_actorIdToPxActorMap.find(actor->GetId());
        if(iit != m_actorIdToPxActorMap.end())
        {
            TBD_FOR(iit->second)
            {
                physx::PxActor* a = *it;
                if(a->isRigidDynamic())
                {
                    physx::PxRigidDynamic* rd = (physx::PxRigidDynamic*)a;
                    rd->addForce(physx::PxVec3(dir.x * newtons, dir.y * newtons, dir.z * newtons));
                }
            }
        }
    }

    VOID PhysX::VApplyTorque(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor) 
    {
        auto iit = m_actorIdToPxActorMap.find(actor->GetId());
        if(iit != m_actorIdToPxActorMap.end())
        {
            TBD_FOR(iit->second)
            {
                physx::PxActor* a = *it;
                if(a->isRigidDynamic())
                {
                    physx::PxRigidDynamic* rd = (physx::PxRigidDynamic*)a;
                    rd->addTorque(physx::PxVec3(dir.x * newtons, dir.y * newtons, dir.z * newtons));
                }
            }
        }
    }

    VOID PhysX::VCreateCharacterController(ActorId id, CONST util::Vec3& pos, FLOAT radius, FLOAT height)
    {
        physx::PxCapsuleControllerDesc desc;
        desc.setToDefault();
        desc.upDirection.y = 1;
        desc.radius = radius;
        desc.height = height;
        desc.stepOffset = 1.0f;

        desc.material = m_materials["default"].m_material;

        physx::PxController* controller = m_pControllerManager->createController(*m_pPhysx, m_pScene, desc);

        controller->setPosition(physx::PxExtendedVec3(pos.x, pos.y, pos.z));

        m_pxActorToActorId[controller->getActor()] = id;
        m_actorIdToPxActorMap[id].push_back(controller->getActor());

        Controller_ c;
        c.m_controller = controller;
        m_controller[id] = c;
    }

    VOID PhysX::VMoveKinematic(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& posDelta, CONST util::Vec3& rotationDelta, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump) 
    {
        auto it = m_controller.find(actor->GetId());
        if(it != m_controller.end())
        {
            Controller_& conroller = it->second;
            
            if(isJump)
            {
                conroller.Jump(posDelta.y);
            }

            conroller.Move(posDelta.x, posDelta.z, deltaMillis);
        }
        else
        {
            physx::PxActor* a = m_actorIdToPxActorMap[actor->GetId()].front(); //Todo: Fix instanced actors? or should they reamain static anyway
            if(a)
            {
                //LOG_ERROR("not yet implemented!");
                if(a->isRigidDynamic())
                {
                    physx::PxRigidDynamic* ad = (physx::PxRigidDynamic*)a;
                    physx::PxTransform trans = ad->getGlobalPose();
                    if(isDeltaMove)
                    {
                        trans.p.x += posDelta.x;
                        trans.p.y += posDelta.y;
                        trans.p.z += posDelta.z;

                        XMVECTOR v1 = XMQuaternionRotationRollPitchYaw(rotationDelta.x, rotationDelta.y, rotationDelta.z);
                        physx::PxQuat q(v1.m128_f32[0], v1.m128_f32[1], v1.m128_f32[2], v1.m128_f32[3]);
                        trans.q *= q;
                    }
                    else
                    {
                        trans.p.x = posDelta.x;
                        trans.p.y = posDelta.y;
                        trans.p.z = posDelta.z;

                        XMVECTOR v1 = XMQuaternionRotationRollPitchYaw(rotationDelta.x, rotationDelta.y, rotationDelta.z);

                        trans.q.x = v1.m128_f32[0];
                        trans.q.y = v1.m128_f32[1];
                        trans.q.z = v1.m128_f32[2];
                        trans.q.w = v1.m128_f32[3];
                    }
                    if(ad->getRigidDynamicFlags() & physx::PxRigidDynamicFlag::eKINEMATIC)
                    {
                        ad->setKinematicTarget(trans);
                    }
                    else
                    {
                        ad->setGlobalPose(trans);
                        ad->clearForce();
                        ad->clearTorque();
                        ad->setLinearVelocity(physx::PxVec3(0,0,0));
                    }
                }
                /*
                if(a->isRigidStatic())
                {
                    physx::PxRigidStatic* ad = (physx::PxRigidStatic*)a;
                    physx::PxTransform trans = ad->getGlobalPose();
                    if(isDeltaMove)
                    {
                        trans.p.x += posDelta.x;
                        trans.p.y += posDelta.y;
                        trans.p.z += posDelta.z;
                    }
                    else
                    {
                        trans.p.x = posDelta.x;
                        trans.p.y = posDelta.y;
                        trans.p.z = posDelta.z;
                    }
                    ad->setGlobalPose(trans);
                }  */
            }
        }
    }

    VOID PhysX::VDebugRender(VOID)
    {

    }

    physx::PxActor* PhysX::AddActor(physx::PxGeometry& geo, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& mat, FLOAT density)
    {

        Material& material = CheckMaterial(mat);

        std::shared_ptr<tbd::TransformComponent> comp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
        util::Vec3 trans = comp->GetTransformation()->GetTranslation() + offsetPosition;;

        physx::PxShape* shape;
        physx::PxActor* pxActor;

        CONST util::Vec4 rot = comp->GetTransformation()->GetRotation();
        physx::PxQuat quat = physx::PxQuat(rot.x, rot.y, rot.z, rot.w);
        quat.normalize();
        physx::PxTransform transform(physx::PxVec3(trans.x, trans.y, trans.z), quat);

        if(material.m_mass > 0)
        {
            physx::PxRigidDynamic* pxDActor = m_pPhysx->createRigidDynamic(transform);
            if(mat == "kinematic")
            {
                pxDActor->setRigidDynamicFlag(physx::PxRigidDynamicFlag::eKINEMATIC, true);
            }
            shape = pxDActor->createShape(geo, *material.m_material);
            pxDActor->setAngularDamping(material.m_angulardamping);
            physx::PxRigidBodyExt::updateMassAndInertia(*pxDActor, density);
            pxActor = pxDActor;
        }
        else
        {
            physx::PxRigidStatic* pxSActor = m_pPhysx->createRigidStatic(transform);
            shape = pxSActor->createShape(geo, *material.m_material);
            pxActor = pxSActor;
        }

        this->m_pScene->addActor(*pxActor);

        this->m_actorIdToPxActorMap[actor->GetId()].push_back(pxActor);
        this->m_pxActorToActorId[pxActor] = actor->GetId();

        return pxActor;
    }

    VOID PhysX::VSyncScene(VOID) 
    {
    
        this->m_pScene->fetchResults(true);

        physx::PxU32 count;
        physx::PxActiveTransform* transforms = m_pScene->getActiveTransforms(count);

        for(physx::PxU32 i = 0; i < count; ++i)
        {
            physx::PxActiveTransform& t = transforms[i];
            ActorId actorid = this->m_pxActorToActorId[t.actor];
        
            std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(actorid);

            if(!actor)
            {
                continue;
            }

            std::shared_ptr<tbd::TransformComponent> comp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();

            comp->GetTransformation()->SetTranslate(t.actor2World.p.x, t.actor2World.p.y, t.actor2World.p.z);
            //comp->GetTransformation()->GetTranslation().Print();
            auto controller = m_controller.find(actorid);

            if(controller == m_controller.end())
            {
                comp->GetTransformation()->SetRotateQuat(t.actor2World.q.x, t.actor2World.q.y,  t.actor2World.q.z, t.actor2World.q.w);
            }
            /*else
            {
                util::Vec3& rot = controller->second.m_rotation;
                //comp->GetTransformation()->SetRotation(rot.x, rot.y, rot.z);
            } */
            event::IEventPtr aMoved(new event::ActorMovedEvent(actor));
            app::g_pApp->GetEventMgr()->VQueueEvent(aMoved);
        }
    
        for(auto it = m_controller.begin(); it != m_controller.end(); ++it)
        {
            it->second.Update(m_lastMillis);
            //it->second.m_controller->move(physx::PxVec3(0, -.025f, 0), 0.0f, 0.5f, physx::PxControllerFilters(), NULL);
        }
    }

    VOID PhysX::VUpdate(FLOAT deltaMillis)
    {
        static float time = 0;
        time += deltaMillis;
        if(time < (FLOAT)(1.0 / 60.0))
        {
            return;
        }
        m_lastMillis = time;
        this->m_pScene->simulate(time);
        time = 0;
    }

    VOID PhysX::ApplyForceTorqueDelegate(event::IEventPtr data)
    {
        if(data->VGetEventType() == event::ApplyForceEvent::TYPE)
        {
            std::shared_ptr<event::ApplyForceEvent> pCastEventData = std::static_pointer_cast<event::ApplyForceEvent>(data);

            VApplyForce(pCastEventData->m_dir, pCastEventData->m_newtons, pCastEventData->m_actor);
        }
        else
        {
            std::shared_ptr<event::ApplyTorqueEvent> pCastEventData = std::static_pointer_cast<event::ApplyTorqueEvent>(data);

            VApplyTorque(pCastEventData->m_torque, pCastEventData->m_newtons, pCastEventData->m_actor);
        }
    }

    VOID PhysX::NewComponentDelegate(event::IEventPtr pEventData) 
    {

        std::shared_ptr<event::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<event::NewComponentCreatedEvent>(pEventData);

        if(pCastEventData->m_id == tbd::PhysicComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(pCastEventData->m_actorId);
            if(!actor)
            {
                LOG_CRITICAL_ERROR("Actor does not exist");
                return;
            }

            std::shared_ptr<tbd::PhysicComponent> physComp = actor->GetComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID).lock();

            if(!physComp)
            {
                return; //should not happen
            }

            std::shared_ptr<tbd::TransformComponent> transComp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();

            if(!transComp)
            {
                LOG_CRITICAL_ERROR("No TrasnformComponent");
            }
            std::shared_ptr<tbd::RenderComponent> renderCmp = actor->GetComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID).lock();
        
            if(renderCmp && !renderCmp->m_instances.empty())
            {
                for(auto it = renderCmp->m_instances.begin(); it != renderCmp->m_instances.end(); ++it)
                {
                    if(physComp->m_shapeType == "box")
                    {
                        this->VCreateCube(physComp->m_dim, actor, *it, physComp->m_material);
                    }
                    else if(physComp->m_shapeType == "sphere")
                    {
                        this->VCreateSphere(physComp->m_radius, actor, *it, physComp->m_material);
                    }
                    else
                    {
                        LOG_CRITICAL_ERROR("ShapeType not implemented");
                    }
                }
            }
            else
            {
                if(physComp->m_shapeType == "plane")
                {
                    this->VCreateStaticPlane(util::Vec3(), actor, physComp->m_material);
                }
                else if(physComp->m_shapeType == "box")
                {
                    this->VCreateCube(physComp->m_dim, actor, util::Vec3(), physComp->m_material);
                }
                else if(physComp->m_shapeType == "sphere")
                {
                    this->VCreateSphere(physComp->m_radius, actor, util::Vec3(), physComp->m_material);
                }
                else if(physComp->m_shapeType == "character")
                {
                    CONST util::Vec3& pos = transComp->GetTransformation()->GetTranslation();
                    this->VCreateCharacterController(actor->GetId(), pos, 0.5f, 1.85f);
                }
                else if(physComp->m_shapeType == "static_mesh")
                {
                    std::shared_ptr<tbd::Mesh> mesh = std::static_pointer_cast<tbd::Mesh>(app::g_pApp->GetCache()->GetHandle(physComp->m_meshFile));
                    VCreateTriangleMesh(actor, mesh.get(), util::Vec3(), physComp->m_material);
                }
                else if(physComp->m_shapeType == "trigger")
                {
                    VCreateTrigger(physComp->m_radius, actor);
                }
                else
                {
                    LOG_CRITICAL_ERROR("ShapeType not implemented");
                }
            }

            physComp->VSetHandled();
        }
    }

    PhysX::~PhysX(VOID)
    {
    
        event::EventListener listener = fastdelegate::MakeDelegate(this, &PhysX::NewComponentDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::NewComponentCreatedEvent::TYPE);

        REMOVE_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, event::ApplyForceEvent::TYPE);
        REMOVE_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, event::ApplyTorqueEvent::TYPE);

        m_actorIdToPxActorMap.clear();

        //TODO controllers/actor release

        /*
        for(auto it = m_controller.begin(); it != m_controller.end(); ++it)
        {
            it->second->release();
        } */
        /*
        for(auto it = m_pxActorToActorId.begin(); it != m_pxActorToActorId.end(); ++it)
        {
            it->first->release();
        } */

        m_pxActorToActorId.clear();

        for(auto it = m_materials.begin(); it != m_materials.end(); ++it)
        {
            it->second.m_material->release();
        }

        delete m_pDesc->cpuDispatcher;

        delete m_pDesc;

        if(m_pDebugConnection)
        {
            m_pDebugConnection->release();
        }

        SAFE_DELETE(m_pDefaultFilterCallback);

        if(m_pScene)
        {
            m_pScene->release();
        }

        if(m_pCooking)
        {
            m_pCooking->release();
        }

        if(m_pControllerManager)
        {
            m_pControllerManager->release();
        }

        if(m_pPhysx)
        {
            m_pPhysx->release();
        }

        if(m_pProfileManager)
        {
            m_pProfileManager->release();
        }
    
        if(m_pFoundation)
        {
            m_pFoundation->release();
        } 
    }

    PhysX::Controller_::Controller_(VOID) : m_controller(NULL), m_time(0), m_jumpDy(1), m_jumpVelo(0), m_a(-9.81f), m_p0(0), m_p1(0), m_maxJumpvalueFunc(1)
    {
        SetJumpSettings(2.0f);
        m_time = 0.55f;
    }

    VOID PhysX::Controller_::Move(FLOAT dx, FLOAT dz, FLOAT deltaMillis)
    {
        physx::PxVec3 dm(dx, 0, dz);
        m_controller->move(dm, 0.0f, deltaMillis, physx::PxControllerFilters(), NULL);
    }

    BOOL PhysX::Controller_::IsOnGround(VOID)
    {
        physx::PxControllerState state;
        m_controller->getState(state);
        return (state.collisionFlags & physx::PxControllerFlag::eCOLLISION_DOWN) == physx::PxControllerFlag::eCOLLISION_DOWN;
    }

    VOID PhysX::Controller_::Jump(FLOAT dy)
    {
        if(IsOnGround())
        {
            m_jumpDy = dy;
            m_time = 0;
            FLOAT th = m_duration * 0.5f;
            m_jumpVelo = -th * m_a;
            m_maxJumpvalueFunc = m_jumpVelo * th + 0.5f * m_a * th * th;
        }
    }

    VOID PhysX::Controller_::SetJumpSettings(FLOAT duration)
    {
        m_duration = duration;
    }

    VOID PhysX::Controller_::Update(FLOAT deltaMillis)
    {
        if(!IsOnGround())
        {
            m_time += deltaMillis;
            //todo: find time bug
            FLOAT p = m_jumpVelo * m_time + 0.5f * m_a * m_time * m_time;
            p = p / m_maxJumpvalueFunc * m_jumpDy;

            m_p0 = m_p1;
            m_p1 = p;
            
            m_controller->move(physx::PxVec3(0, m_p1 - m_p0, 0), 0.0f, deltaMillis, physx::PxControllerFilters(), NULL);
        }
        else
        {
            m_p0 = 0;
            m_p1 = 0;
            m_jumpVelo = 0;
            m_jumpDy = 1;
            m_time = 0;
            m_maxJumpvalueFunc = 1;
        }
    }
};