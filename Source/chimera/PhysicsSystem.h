#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "Event.h"
#include "Vec3.h"

namespace tbd
{
    class Mesh;
}

namespace logic 
{
    class IPhysicsSystem
    {
    public:

        virtual BOOL VInit(VOID) = 0;
    
        virtual VOID VCreateStaticPlane(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, std::string& material) = 0;

        virtual VOID VCreateSphere(FLOAT radius, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material) = 0;

        virtual VOID VCreateCube(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material) = 0;

        virtual VOID VCreateTrigger(FLOAT radius, std::shared_ptr<tbd::Actor> actor) = 0;

        virtual VOID VCreateTriangleMesh(std::shared_ptr<tbd::Actor> actor, CONST tbd::Mesh* mesh, CONST util::Vec3& offsetPosition, std::string& material) = 0;

        virtual VOID VCreateCharacterController(ActorId id, CONST util::Vec3& pos, FLOAT radius, FLOAT height) = 0;

        virtual VOID VRemoveActor(ActorId id) = 0;

        virtual VOID VApplyForce(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor) = 0;

        virtual VOID VApplyTorque(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor) = 0;

        virtual VOID VMoveKinematic(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3* pos, CONST util::Vec4* quat, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE) = 0;

        virtual VOID VMoveKinematic(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3* pos, CONST util::Vec3* axis, FLOAT angle, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE) = 0;

        virtual VOID VDebugRender(VOID) { }

        virtual VOID VSyncScene(VOID) = 0;

        virtual VOID VUpdate(FLOAT deltaMillis) = 0;

        virtual ~IPhysicsSystem(VOID) {}
    };

    class IPhysicsDebugRenderer 
    {
    public:
        virtual VOID VDrawLine(CONST util::Vec3& start, CONST util::Vec3& end) = 0;
        virtual VOID VDrawCube(CONST util::Vec3& extends, CONST util::Vec3& position) = 0;
    };

    class ErrorCallback : public physx::PxErrorCallback
    {
    public:
        ErrorCallback(VOID) {}
        VOID reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line) {
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
        ~ErrorCallback(VOID) {}
    };

    class Allocator : public physx::PxAllocatorCallback
    {
    public:
        Allocator(VOID) {}
        void* allocate(size_t size, const char* typeName, const char* filename, int line) {
            return _aligned_malloc(size, 16);
        }
        VOID deallocate(VOID* ptr) {
            _aligned_free(ptr);
        }
        ~Allocator(VOID) {}
    };

    struct Material 
    {
        FLOAT m_staticFriction;
        FLOAT m_dynamicFriction;
        FLOAT m_restitution;
        FLOAT m_mass;
        FLOAT m_angulardamping;
        //FLOAT m_
        physx::PxMaterial* m_material;
        Material(FLOAT staticFriction, FLOAT dynamicFriction, FLOAT restitution, FLOAT mass, FLOAT angularDamping, physx::PxPhysics* physx) : 
        m_staticFriction(staticFriction), m_dynamicFriction(dynamicFriction), m_restitution(restitution), m_mass(mass), m_angulardamping(angularDamping) {

            m_material = physx->createMaterial(staticFriction, dynamicFriction, restitution);
        }

        Material(VOID) : m_material(NULL) {}

        ~Material(VOID) { }
    };

    class PhysX : public IPhysicsSystem 
    {
        friend class DefaultFilterCallback;

        class Controller_
        {
        private:
            FLOAT m_jumpDy;
            FLOAT m_time;
            FLOAT m_duration;
            FLOAT m_p0;
            FLOAT m_p1;
            FLOAT m_maxJumpvalueFunc;
            FLOAT m_jumpVelo;
            FLOAT m_a;

        public:
            Controller_(VOID);
            VOID SetJumpSettings(FLOAT duration);
            VOID Update(FLOAT deltaMillis);
            VOID Move(FLOAT dx, FLOAT dy, FLOAT deltaMilli);
            VOID Jump(FLOAT dy);
            BOOL IsOnGround(VOID);
            physx::PxController* m_controller;
            util::Vec3 m_rotation;
        };

    private:
        physx::PxFoundation* m_pFoundation;
        physx::PxPhysics* m_pPhysx;
        physx::PxProfileZoneManager* m_pProfileManager;
        physx::PxCooking* m_pCooking;
        physx::PxScene* m_pScene;
        physx::PxSceneDesc* m_pDesc;
        physx::PxControllerManager* m_pControllerManager;
        std::map<ActorId, Controller_> m_controller;
        std::map<ActorId, std::vector<physx::PxActor*>> m_actorIdToPxActorMap;
        std::map<physx::PxActor*, ActorId> m_pxActorToActorId;

        DefaultFilterCallback* m_pDefaultFilterCallback;
        ErrorCallback m_errorCallback;
        Allocator m_allocator;

        FLOAT m_lastMillis;

        PVD::PvdConnection* m_pDebugConnection;

        physx::PxActor* AddActor(physx::PxGeometry& geo, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material, FLOAT density);

        Material& CheckMaterial(std::string material) {

            auto it = this->m_materials.find(material);

            if(it == m_materials.end())
            {
                LOG_CRITICAL_ERROR_A("Material: '%s' not found", material.c_str());
            }

            return it->second;
        }

        std::map<std::string, Material> m_materials;

    public:

        PhysX(VOID);

        BOOL VInit(VOID);

        VOID VCreateStaticPlane(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, std::string& material);

        VOID VCreateSphere(FLOAT radius, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material);

        VOID VCreateCube(CONST util::Vec3& dimension, std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& offsetPosition, std::string& material);

        VOID VCreateTriangleMesh(std::shared_ptr<tbd::Actor> actor, CONST tbd::Mesh* mesh, CONST util::Vec3& position, std::string& material);

        VOID VCreateTrigger(FLOAT radius, std::shared_ptr<tbd::Actor> actor);

        VOID VRemoveActor(ActorId id);

        VOID VApplyForce(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor);

        VOID VApplyTorque(CONST util::Vec3& dir, FLOAT newtons, std::shared_ptr<tbd::Actor> actor);

        VOID VCreateCharacterController(ActorId id, CONST util::Vec3& pos, FLOAT radius, FLOAT height);

        VOID VMoveKinematic(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3* pos, CONST util::Vec4* quat, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE);

        VOID VMoveKinematic(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3* pos, CONST util::Vec3* axis, FLOAT angle, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE);

        VOID VDebugRender(VOID);

        VOID VSyncScene(VOID);

        VOID VUpdate(FLOAT deltaMillis);

        VOID NewComponentDelegate(event::IEventPtr data);

        VOID ApplyForceTorqueDelegate(event::IEventPtr data);

        //TODO VOID ComponentRemovedDelegate(event::IEventPtr data);

        ~PhysX(VOID);
    };
};