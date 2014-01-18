#include "PhysicsSystem.h"
#include "Components.h"
#include "Mesh.h"
#include "Event.h"

#ifdef free
    #define free_save free
    #undef free
#endif

#ifdef realloc
    #define realloc_save realloc
    #undef realloc
#endif

#include <physx/PxPhysicsAPI.h>
#include "physx/PxToolkit.h"
#include "physx/pvd/PxVisualDebugger.h"

#ifdef free_save
    #define free free_save
#endif

#ifdef realloc_save
    #define realloc realloc_save
#endif


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

namespace chimera 
{
    using namespace physx;

    namespace px
    {
        Material::Material(float staticFriction, float dynamicFriction, float restitution, float mass, float angularDamping, physx::PxPhysics* physx) : 
        m_staticFriction(staticFriction), m_dynamicFriction(dynamicFriction), m_restitution(restitution), m_mass(mass), m_angulardamping(angularDamping) {
            m_material = physx->createMaterial(staticFriction, dynamicFriction, restitution);
        }
    }

    class ErrorCallback : public physx::PxErrorCallback
    {
    public:
        ErrorCallback(void) {}
        void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line) {
            char* error = NULL;
            switch(code) {
            case physx::PxErrorCode::eNO_ERROR : { return; } break;
            case physx::PxErrorCode::eDEBUG_INFO : 
                { 
                    error = "Debug Info";                         
                } break;

            case physx::PxErrorCode::eDEBUG_WARNING : 
                {
                    error = "Debug Warning";
                } break;

            case physx::PxErrorCode::eINVALID_PARAMETER : 
                {
                    error = "Invalid Parameter";                            
                } break;
            case physx::PxErrorCode::eINVALID_OPERATION : 
                {
                    error = "Invalid Operation";
                } break;
            case physx::PxErrorCode::eOUT_OF_MEMORY : 
                {
                    error = "Out of Memory";
                } break;
            case physx::PxErrorCode::eINTERNAL_ERROR : 
                {
                    error = "Internal Error";
                } break;
            case physx::PxErrorCode::eMASK_ALL : 
                {
                    error = "Mask All";
                } break;

            default:
                {
                    error = "Default Error";
                } break;
            }
            std::string errorMessage("PHX: ");
            errorMessage += message;
            errorMessage += " : " + std::string(error) + "\n";
            OutputDebugStringA(errorMessage.c_str());
            //std::cout << buffer << std::endl;
        }
        ~ErrorCallback(void) {}
    };

    class Allocator : public physx::PxAllocatorCallback
    {
    public:
        Allocator(void) {}
        void* allocate(size_t size, const char* typeName, const char* filename, int line) {
            return _aligned_malloc(size, 16);
        }
        void deallocate(void* ptr) {
            _aligned_free(ptr);
        }
        ~Allocator(void) {}
    };

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
        pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_CONTACT_POINTS;

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
        void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)
        {

        }
        void onWake(PxActor** actors, PxU32 count)
        {

        }
        void onSleep(PxActor** actors, PxU32 count)
        {

        }
        void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs)
        {
            for(PxU32 i = 0; i < nbPairs; ++i)
            {
                const PxContactPair& cp = pairs[i];

                if(cp.events & PxPairFlag::eNOTIFY_TOUCH_FOUND)
                {
                    PxContactPairPoint* pts = new PxContactPairPoint[cp.contactCount];
                    cp.extractContacts(pts, cp.contactCount * sizeof(PxContactPairPoint));
                    ActorId id0 = m_pPhysix->m_pxActorToActorId[pairHeader.actors[0]];
                    ActorId id1 = m_pPhysix->m_pxActorToActorId[pairHeader.actors[1]];
                    CollisionEvent* ce = new CollisionEvent();
                    ce->m_actor0 = id0;
                    ce->m_actor1 = id1;
                    ce->m_impulse.Set(pts[0].impulse.x, pts[0].impulse.y, pts[0].impulse.z);
                    ce->m_position.Set(pts[0].position.x, pts[0].position.y, pts[0].position.z);
                    QUEUE_EVENT(ce);
                    SAFE_ARRAY_DELETE(pts);
                }
            }
        }

        void onTrigger(PxTriggerPair* pairs, PxU32 count)
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
                chimera::TriggerEvent* te = new chimera::TriggerEvent();
                te->m_triggerActor = id0;
                te->m_didTriggerActor = id1;
                QUEUE_EVENT(te);
            }
        }
    };


    PhysX::PhysX(void) : m_pCooking(NULL), m_pDesc(NULL), m_pFoundation(NULL), m_pPhysx(NULL), m_pProfileManager(NULL), m_pScene(NULL), m_pControllerManager(NULL), m_pDebugConnection(NULL)
    {
        m_pDefaultFilterCallback = new DefaultFilterCallback(this);
    }

    Allocator m_allocator;
    ErrorCallback m_errorCallback;

    bool PhysX::VInit(void) 
    {
        m_pFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, m_allocator, m_errorCallback);
        if(!m_pFoundation)
        {
            return false;
        }

        m_pProfileManager = &physx::PxProfileZoneManager::createProfileZoneManager(m_pFoundation);
        if(!m_pProfileManager)
        {
            return false;
        }

        bool recordMemoryAllocations = false;
    
    #ifdef _DEBUG
        recordMemoryAllocations = true;
    #endif

        m_pPhysx = PxCreatePhysics(PX_PHYSICS_VERSION, *m_pFoundation, physx::PxTolerancesScale(), recordMemoryAllocations, m_pProfileManager);
        if(!m_pPhysx)
        {
            return false;
        }

        m_pControllerManager = PxCreateControllerManager(*m_pFoundation);
        RETURN_IF_FAILED(m_pControllerManager);

        m_pCooking = PxCreateCooking(PX_PHYSICS_VERSION, *m_pFoundation, physx::PxCookingParams());
        if(!m_pCooking)
        {
            return false;
        }

        m_pDesc = new physx::PxSceneDesc(this->m_pPhysx->getTolerancesScale());
        m_pDesc->gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);

        m_pDesc->filterShader = DefaultFilterShader;//physx::PxDefaultSimulationFilterShader;
  
        m_pDesc->cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(4);

        m_pDesc->flags |= physx::PxSceneFlag::eENABLE_ACTIVETRANSFORMS | physx::PxSceneFlag::eENABLE_KINEMATIC_PAIRS | physx::PxSceneFlag::eENABLE_KINEMATIC_STATIC_PAIRS;// | physx::PxSceneFlag::eENABLE_SWEPT_INTEGRATION;

        m_pScene = this->m_pPhysx->createScene(*m_pDesc);

        if(!m_pScene)
        {
            return false;
        }

        m_pScene->setSimulationEventCallback(m_pDefaultFilterCallback);

        /*if(!PxInitExtensions(*m_pPhysx))
        {
            return FALSE;
        } */

        //set default materials
        m_materials["dynamic"] = px::Material(0.5f, 0.5f, 0.1f, 1.0f, 1.0f, m_pPhysx);
        m_materials["static"] = px::Material(0.5f, 0.5f, 0.1f, 0.0f, 1.0f, m_pPhysx);
        m_materials["bouncy"] = px::Material(0.5f, 0.5f, 0.8f, 1.0f, 2.0f, m_pPhysx);
        m_materials["kinematic"] = px::Material(0.5f, 0.5f, 0.8f, 1.0f, 2.0f, m_pPhysx);
        m_materials["default"] = px::Material(0.5f, 0.5f, 0.1f, 1.0f, 2.0f, m_pPhysx);
        m_materials[""] = px::Material(0.5f, 0.5f, 0.1f, 1.0f, 2.0f, m_pPhysx);
        //m_materials["bouncy"].m_material->setFlag(physx::PxMaterialFlag::eDISABLE_STRONG_FRICTION, FALSE);
        //end

        ADD_EVENT_LISTENER(this, &PhysX::NewComponentDelegate, CM_EVENT_COMPONENT_CREATED);
        ADD_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, CM_EVENT_APPLY_FORCE);
        ADD_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, CM_EVENT_APPLY_TORQUE);

    #if 0
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


        return true;
    }

    void PhysX::VCreateStaticPlane(const util::Vec3& dimension, IActor* actor, std::string& material) 
    {
        physx::PxMaterial* mat = m_materials[material].m_material;
        if(!mat)
        {
            mat = m_materials["default"].m_material;
        }
        TransformComponent* comp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);
        util::Vec3 trans = comp->GetTransformation()->GetTranslation();
        physx::PxRigidStatic* plane = PxCreatePlane(*m_pPhysx, physx::PxPlane(physx::PxVec3(trans.x, trans.y, trans.z), physx::PxVec3(0, 1, 0)), *mat);

        m_pScene->addActor(*plane);

        m_actorIdToPxActorMap[actor->GetId()].push_back(plane);
        m_pxActorToActorId[plane] = actor->GetId();
    }

    void PhysX::VCreateSphere(float radius, IActor* actor, const util::Vec3& offsetPosition, std::string& material) 
    {
        physx::PxSphereGeometry geo(radius);
        this->AddActor(geo, actor, offsetPosition, material, physx::PxPi * 4.0f/3.0f * radius * radius * radius);
    }

    void PhysX::VCreateCube(const util::Vec3& dimension, IActor* actor, const util::Vec3& offsetPosition, std::string& material) 
    {
        physx::PxBoxGeometry geo(dimension.x * 0.5f, dimension.y * 0.5f, dimension.z * 0.5f);
        this->AddActor(geo, actor, offsetPosition, material, dimension.x * dimension.y * dimension.z);
    }

    void PhysX::CreateTriangleConcaveMesh(IActor* actor, const IMesh* mesh, const util::Vec3& offsetPosition, std::string& material)
    {
        physx::PxTriangleMeshDesc meshDesc;

        uint stride = mesh->VGetVertexStride() / sizeof(uint);
        uint count = mesh->VGetVertexCount();

        physx::PxVec3* verts = new physx::PxVec3[mesh->VGetVertexCount()];

        for(uint i = 0; i < mesh->VGetVertexCount(); ++i)
        {
            physx::PxVec3 v;
            v.x = mesh->VGetVertices()[i * stride + 0];
            v.y = mesh->VGetVertices()[i * stride + 1];
            v.z = mesh->VGetVertices()[i * stride + 2];
            verts[i] = v;
        }
        meshDesc.points.count = count;
        meshDesc.points.stride = sizeof(physx::PxVec3);
        meshDesc.points.data = verts;

        physx::PxU32* indices32 = new physx::PxU32[mesh->VGetIndexCount()];
        uint index = 0;
        for(uint i = 0; i < mesh->VGetIndexCount() / 3; ++i)
        {
            uint i0 = mesh->VGetIndices()[3 * i + 0];
            uint i1 = mesh->VGetIndices()[3 * i + 1];
            uint i2 = mesh->VGetIndices()[3 * i + 2];
            indices32[3 * i + 0] = i0;
            indices32[3 * i + 1] = i2;
            indices32[3 * i + 2] = i1;
        }

        meshDesc.triangles.count = mesh->VGetIndexCount() / 3;
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
        scale.scale.z = scale.scale.y = scale.scale.x = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM)->GetTransformation()->GetScale().x;
        physx::PxTriangleMeshGeometry geo(pxmesh, scale);

        AddActor(geo, actor, offsetPosition, material, 1);

        SAFE_ARRAY_DELETE(indices32);
        SAFE_ARRAY_DELETE(verts);
    }

    void PhysX::CreateTriangleConvexMesh(IActor* actor, const IMesh* mesh, const util::Vec3& offsetPosition, std::string& material)
    {
        physx::PxConvexMeshDesc meshDesc;

        uint stride = mesh->VGetVertexStride() / sizeof(uint);
        uint count = mesh->VGetVertexCount();

        physx::PxVec3* verts = new physx::PxVec3[mesh->VGetVertexCount()];

        for(uint i = 0; i < mesh->VGetVertexCount(); ++i)
        {
            physx::PxVec3 v;
            v.x = mesh->VGetVertices()[i * stride + 0];
            v.y = mesh->VGetVertices()[i * stride + 1];
            v.z = mesh->VGetVertices()[i * stride + 2];
            verts[i] = v;
        }
        meshDesc.points.count = count;
        meshDesc.points.stride = sizeof(physx::PxVec3);
        meshDesc.points.data = verts;
        meshDesc.flags = physx::PxConvexFlag::eCOMPUTE_CONVEX;

        physx::PxU32* indices32 = new physx::PxU32[mesh->VGetIndexCount()];
        uint index = 0;
        for(uint i = 0; i < mesh->VGetIndexCount() / 3; ++i)
        {
            uint i0 = mesh->VGetIndices()[3 * i + 0];
            uint i1 = mesh->VGetIndices()[3 * i + 1];
            uint i2 = mesh->VGetIndices()[3 * i + 2];
            indices32[3 * i + 0] = i0;
            indices32[3 * i + 1] = i2;
            indices32[3 * i + 2] = i1;
        }

        meshDesc.triangles.count = mesh->VGetIndexCount() / 3;
        meshDesc.triangles.stride = 3 * sizeof(physx::PxU32);
        meshDesc.triangles.data = indices32;

        PxToolkit::MemoryOutputStream writeBuffer;
        if(!m_pCooking->cookConvexMesh(meshDesc, writeBuffer))
        {
            LOG_CRITICAL_ERROR("failed to cook triangle mesh");
        }

        PxToolkit::MemoryInputData readBuffer(writeBuffer.getData(), writeBuffer.getSize());
        physx::PxConvexMesh* pxmesh = m_pPhysx->createConvexMesh(readBuffer);
        physx::PxMeshScale scale;
        scale.scale.z = scale.scale.y = scale.scale.x = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM)->GetTransformation()->GetScale().x;
        physx::PxConvexMeshGeometry geo(pxmesh, scale);

        AddActor(geo, actor, offsetPosition, material, 1);

        SAFE_ARRAY_DELETE(indices32);
        SAFE_ARRAY_DELETE(verts);
    }

    void PhysX::VCreateTriangleMesh(IActor*actor, const IMesh* mesh, const util::Vec3& offsetPosition, std::string& material, std::string& shapeType)
    {
        if(shapeType == "convex")
        {
            CreateTriangleConvexMesh(actor, mesh, offsetPosition, material);
        }
        else if(shapeType == "concave")
        {
            CreateTriangleConcaveMesh(actor, mesh, offsetPosition, material);
        }
        else
        {
            LOG_CRITICAL_ERROR("shapetype not implemented!");
        }
    }

    void PhysX::VCreateTrigger(float radius, IActor* actor) 
    {
        physx::PxSphereGeometry geo(radius);
        std::string m("static");
        physx::PxActor* a = AddActor(geo, actor, util::Vec3(0,0,0), m, physx::PxPi * 4.0f/3.0f * radius * radius * radius);
        physx::PxShape* shape;
        ((physx::PxRigidStatic*)a)->getShapes(&shape, 1);
        shape->setFlag(PxShapeFlag::eTRIGGER_SHAPE, true);
    }

    void PhysX::VRemoveActor(ActorId id) 
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

    void PhysX::VApplyForce(const util::Vec3& dir, float newtons, IActor* actor) 
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

    void PhysX::VApplyTorque(const util::Vec3& dir, float newtons, IActor* actor) 
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

    void PhysX::VCreateCharacterController(ActorId id, const util::Vec3& pos, float radius, float height)
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

    void ScaleGeometry(float scale,  physx::PxShape* shape)
    {
        switch(shape->getGeometryType())
        {
        case physx::PxGeometryType::eBOX : 
            {
                physx::PxBoxGeometry geo;
                shape->getBoxGeometry(geo);
                geo.halfExtents = physx::PxVec3(scale);
                shape->setGeometry(geo);
            } break;
        case physx::PxGeometryType::eSPHERE : 
            {
                physx::PxSphereGeometry geo;
                shape->getSphereGeometry(geo);
                geo.radius = scale;
                shape->setGeometry(geo);

            } break;
        case physx::PxGeometryType::eCONVEXMESH : 
            {
                physx::PxConvexMeshGeometry geo;
                shape->getConvexMeshGeometry(geo);
                geo.scale = physx::PxMeshScale(physx::PxVec3(scale), physx::PxQuat(0,0,0,1));
                shape->setGeometry(geo);
            } break;
        }
    }

    void PhysX::VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec3* axis, float angle, float deltaMillis, bool isDeltaMove, bool isJump) 
    {
        XMVECTOR quat = XMQuaternionRotationNormal(XMLoadFloat3(&axis->m_v), angle);
        util::Vec4 q;
        XMStoreFloat4(&q.m_v, quat);
        VMoveKinematic(actor, pos, &q, deltaMillis, isDeltaMove, isJump);
    }

    void PhysX::VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec4* quat, float deltaMillis, bool isDeltaMove, bool isJump)
    {
        auto it = m_controller.find(actor->GetId());
        if(it != m_controller.end() && pos)
        {
            Controller_& conroller = it->second;
            
            if(pos)
            {
                if(isJump)
                {
                    conroller.Jump(pos->y);
                }

                conroller.Move(pos->x, pos->z, deltaMillis);
            }
        }
        else
        {
            physx::PxActor* a = m_actorIdToPxActorMap[actor->GetId()].front(); //Todo: Fix instanced actors? or should they remain static anyway
            if(a)
            {
                if(a->isRigidDynamic())
                {
                    physx::PxRigidDynamic* ad = (physx::PxRigidDynamic*)a;
                    physx::PxTransform trans = ad->getGlobalPose();
                    if(isDeltaMove)
                    {
                        if(pos)
                        {
                            trans.p.x += pos->x;
                            trans.p.y += pos->y;
                            trans.p.z += pos->z;
                        }

                        if(quat)
                        {
                            physx::PxQuat q(quat->x, quat->y, quat->z, quat->w);
                            trans.q *= q;
                        }
                    }
                    else
                    {
                        if(pos)
                        {
                            trans.p.x = pos->x;
                            trans.p.y = pos->y;
                            trans.p.z = pos->z;
                        }

                        if(quat)
                        {
                            trans.q.x = quat->x;
                            trans.q.y = quat->y;
                            trans.q.z = quat->z;
                            trans.q.w = quat->w;
                        }
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

                    float scale = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM)->GetTransformation()->GetScale().x;
                    uint shapeCount = ad->getNbShapes();
                    physx::PxShape** shapes = new physx::PxShape*[shapeCount];
                    ad->getShapes(shapes, shapeCount, 0);
                    for(uint i = 0; i < shapeCount; ++i)
                    {
                        physx::PxShape* shape = shapes[i];
                        ScaleGeometry(scale, shape);
                    }
                    SAFE_ARRAY_DELETE(shapes);
                }
            }
        }
    }

    void PhysX::VDebugRender(void)
    {

    }

    physx::PxActor* PhysX::AddActor(physx::PxGeometry& geo, IActor* actor, const util::Vec3& offsetPosition, std::string& mat, float density)
    {
        px::Material& material = CheckMaterial(mat);

        TransformComponent* comp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);
        util::Vec3 trans = comp->GetTransformation()->GetTranslation() + offsetPosition;;

        physx::PxShape* shape;
        physx::PxActor* pxActor;

        const util::Vec4 rot = comp->GetTransformation()->GetRotation();
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

    void PhysX::VSyncScene(void) 
    {
    
        this->m_pScene->fetchResults(true);

        physx::PxU32 count;
        physx::PxActiveTransform* transforms = m_pScene->getActiveTransforms(count);

        for(physx::PxU32 i = 0; i < count; ++i)
        {
            physx::PxActiveTransform& t = transforms[i];
            ActorId actorid = this->m_pxActorToActorId[t.actor];
        
            IActor* actor = CmGetApp()->VGetLogic()->VFindActor(actorid);

            if(!actor)
            {
                continue;
            }

            TransformComponent* comp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);

            comp->GetTransformation()->SetTranslate(t.actor2World.p.x, t.actor2World.p.y, t.actor2World.p.z);
            //comp->GetTransformation()->GetTranslation().Print();
            auto controller = m_controller.find(actorid);

            if(controller == m_controller.end())
            {
                //const physx::PxF32 eps = 0.001f;
                //if(t.actor2World.q.x > eps || t.actor2World.q.y > eps || t.actor2World.q.z > eps || t.actor2World.q.w > eps)
                {
                    comp->GetTransformation()->SetRotateQuat(t.actor2World.q.x, t.actor2World.q.y,  t.actor2World.q.z, t.actor2World.q.w);
                }
            }
            /*else
            {
                util::Vec3& rot = controller->second.m_rotation;
                //comp->GetTransformation()->SetRotation(rot.x, rot.y, rot.z);
            } */
            QUEUE_EVENT(new chimera::ActorMovedEvent(actor));
        }
    
        for(auto it = m_controller.begin(); it != m_controller.end(); ++it)
        {
            it->second.Update(m_lastMillis);
            //it->second.m_controller->move(physx::PxVec3(0, -.025f, 0), 0.0f, 0.5f, physx::PxControllerFilters(), NULL);
        }
    }

    void PhysX::VUpdate(float deltaMillis)
    {
        static float time = 0;
        time += deltaMillis;
        if(time < (float)(1.0 / 60.0))
        {
            return;
        }
        m_lastMillis = time;//min(1.0 / 60.0, time);
        this->m_pScene->simulate(time);
        time = 0;
    }

    void PhysX::ApplyForceTorqueDelegate(IEventPtr data)
    {
        if(data->VGetEventType() == CM_EVENT_APPLY_FORCE)
        {
            std::shared_ptr<ApplyForceEvent> pCastEventData = std::static_pointer_cast<ApplyForceEvent>(data);

            VApplyForce(pCastEventData->m_dir, pCastEventData->m_newtons, pCastEventData->m_actor);
        }
        else
        {
            std::shared_ptr<ApplyTorqueEvent> pCastEventData = std::static_pointer_cast<ApplyTorqueEvent>(data);

            VApplyTorque(pCastEventData->m_torque, pCastEventData->m_newtons, pCastEventData->m_actor);
        }
    }

    void PhysX::CreateFromActor(IActor* actor)
    {
        PhysicComponent* physComp = GetActorCompnent<PhysicComponent>(actor, CM_CMP_PHX);
        if(!actor)
        {
            LOG_CRITICAL_ERROR("Actor does not exist");
            return;
        }

        if(!physComp)
        {
            return; //should not happen
        }

        TransformComponent* transComp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);

        if(!transComp)
        {
            LOG_CRITICAL_ERROR("No TrasnformComponent");
        }
        RenderComponent* renderCmp = GetActorCompnent<RenderComponent>(actor, CM_CMP_RENDERING);

        if(renderCmp && !renderCmp->m_instances.empty())
        {
            for(auto it = renderCmp->m_instances.begin(); it != renderCmp->m_instances.end(); ++it)
            {
                if(physComp->m_shapeStyle == "box")
                {
                    this->VCreateCube(physComp->m_dim, actor, *it, physComp->m_material);
                }
                else if(physComp->m_shapeStyle == "sphere")
                {
                    this->VCreateSphere(physComp->m_radius, actor, *it, physComp->m_material);
                }
                else
                {
                    LOG_CRITICAL_ERROR("ShapeStyle not implemented");
                }
            }
        }
        else
        {
            if(physComp->m_shapeStyle == "plane")
            {
                this->VCreateStaticPlane(util::Vec3(), actor, physComp->m_material);
            }
            else if(physComp->m_shapeStyle == "box")
            {
                this->VCreateCube(physComp->m_dim, actor, util::Vec3(), physComp->m_material);
            }
            else if(physComp->m_shapeStyle == "sphere")
            {
                this->VCreateSphere(physComp->m_radius, actor, util::Vec3(), physComp->m_material);
            }
            else if(physComp->m_shapeStyle == "character")
            {
                const util::Vec3& pos = transComp->GetTransformation()->GetTranslation();
                this->VCreateCharacterController(actor->GetId(), pos, 0.5f, 1.85f);
            }
            else if(physComp->m_shapeStyle == "mesh")
            {
                std::shared_ptr<IMeshSet> meshSet = std::static_pointer_cast<IMeshSet>(CmGetApp()->VGetCache()->VGetHandle(physComp->m_meshFile));
                IMesh* mesh = meshSet->VGetMesh(0);
                VCreateTriangleMesh(actor, mesh, util::Vec3(), physComp->m_material, physComp->m_shapeType);
                m_resourceToActor[meshSet->VGetResource().m_name] = actor->GetId();
            }
            else if(physComp->m_shapeStyle == "trigger")
            {
                VCreateTrigger(physComp->m_radius, actor);
            }
            else
            {
                LOG_CRITICAL_ERROR("ShapeStyle not implemented");
            }
        }
    }

    void PhysX::NewComponentDelegate(chimera::IEventPtr pEventData) 
    {

        std::shared_ptr<chimera::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<chimera::NewComponentCreatedEvent>(pEventData);

        if(pCastEventData->m_id == CM_CMP_PHX)
        {
            IActor* actor = CmGetApp()->VGetLogic()->VFindActor(pCastEventData->m_actorId);
            CreateFromActor(actor);
        }
    }

    void PhysX::OnResourceChanged(IEventPtr data)
    {
        std::shared_ptr<ResourceChangedEvent> event = std::static_pointer_cast<ResourceChangedEvent>(data);

        auto it = m_resourceToActor.find(event->m_resource);
        if(it != m_resourceToActor.end())
        {
            VRemoveActor(it->second);
            IActor* actor = CmGetApp()->VGetLogic()->VFindActor(it->second);
            CreateFromActor(actor);
        }
    }

    PhysX::~PhysX(void)
    {
        REMOVE_EVENT_LISTENER(this, &PhysX::NewComponentDelegate, CM_EVENT_COMPONENT_CREATED);
        REMOVE_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, CM_EVENT_APPLY_FORCE);
        REMOVE_EVENT_LISTENER(this, &PhysX::ApplyForceTorqueDelegate, CM_EVENT_APPLY_TORQUE);

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
            if(it->second.m_material)
            {
                it->second.m_material->release();
            }
        }

        delete m_pDesc->cpuDispatcher;

        delete m_pDesc;

        /*if(m_pDebugConnection)
        {
            m_pDebugConnection->release();
        }*/

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

    PhysX::Controller_::Controller_(void) : m_controller(NULL), m_time(0), m_jumpDy(1), m_jumpVelo(0), m_a(-9.81f), m_p0(0), m_p1(0), m_maxJumpvalueFunc(1)
    {
        SetJumpSettings(2.0f);
        m_time = 0.55f;
    }

    void PhysX::Controller_::Move(float dx, float dz, float deltaMillis)
    {
        physx::PxVec3 dm(dx, 0, dz);
        m_controller->move(dm, 0.0f, deltaMillis, physx::PxControllerFilters(), NULL);
    }

    bool PhysX::Controller_::IsOnGround(void)
    {
        physx::PxControllerState state;
        m_controller->getState(state);
        return (state.collisionFlags & physx::PxControllerFlag::eCOLLISION_DOWN) == physx::PxControllerFlag::eCOLLISION_DOWN;
    }

    void PhysX::Controller_::Jump(float dy)
    {
        if(IsOnGround())
        {
            m_jumpDy = dy;
            m_time = 0;
            float th = m_duration * 0.5f;
            m_jumpVelo = -th * m_a;
            m_maxJumpvalueFunc = m_jumpVelo * th + 0.5f * m_a * th * th;
        }
    }

    void PhysX::Controller_::SetJumpSettings(float duration)
    {
        m_duration = duration;
    }

    void PhysX::Controller_::Update(float deltaMillis)
    {
        if(!IsOnGround())
        {
            m_time += deltaMillis;
            //todo: find time bug
            float p = m_jumpVelo * m_time + 0.5f * m_a * m_time * m_time;
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