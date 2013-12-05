#pragma once
#include "stdafx.h"
#include "Vec3.h"

namespace physx
{
    class PxMaterial;
    class PxPhysics;
    class PxController;
    class PxFoundation;
    class PxProfileZoneManager;
    class PxCooking;
    class PxScene;
    class PxSceneDesc;
    class PxControllerManager;
    class PxActor;
    class PxGeometry;
};

namespace PVD
{
    class PvdConnection;
};

namespace chimera 
{
    class Mesh;
    class ResHandle;
    class PhysicComponent;
    class IPhysicsSystem
    {
    public:

        virtual BOOL VInit(VOID) = 0;
    
        virtual VOID VCreateStaticPlane(CONST util::Vec3& dimension, IActor* actor, std::string& material) = 0;

        virtual VOID VCreateSphere(FLOAT radius, IActor* actor, CONST util::Vec3& offsetPosition, std::string& material) = 0;

        virtual VOID VCreateCube(CONST util::Vec3& dimension, IActor* actor, CONST util::Vec3& offsetPosition, std::string& material) = 0;

        virtual VOID VCreateTrigger(FLOAT radius, IActor* actor) = 0;

        virtual VOID VCreateTriangleMesh(IActor* actor, CONST IMesh* mesh, CONST util::Vec3& offsetPosition, std::string& material, std::string & shapeType) = 0;

        virtual VOID VCreateCharacterController(ActorId id, CONST util::Vec3& pos, FLOAT radius, FLOAT height) = 0;

        virtual VOID VRemoveActor(ActorId id) = 0;

        virtual VOID VApplyForce(CONST util::Vec3& dir, FLOAT newtons, IActor* actor) = 0;

        virtual VOID VApplyTorque(CONST util::Vec3& dir, FLOAT newtons, IActor* actor) = 0;

        virtual VOID VMoveKinematic(IActor* actor, CONST util::Vec3* pos, CONST util::Vec4* quat, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE) = 0;

        virtual VOID VMoveKinematic(IActor* actor, CONST util::Vec3* pos, CONST util::Vec3* axis, FLOAT angle, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE) = 0;

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

    namespace px
    {
        struct Material 
        {
            FLOAT m_staticFriction;
            FLOAT m_dynamicFriction;
            FLOAT m_restitution;
            FLOAT m_mass;
            FLOAT m_angulardamping;
            //FLOAT m_
            physx::PxMaterial* m_material;
            Material(FLOAT staticFriction, FLOAT dynamicFriction, FLOAT restitution, FLOAT mass, FLOAT angularDamping, physx::PxPhysics* physx);

            Material(VOID) : m_material(NULL) {}

            ~Material(VOID) { }
        };
    }


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
        std::map<std::string, ActorId> m_resourceToActor;

        DefaultFilterCallback* m_pDefaultFilterCallback;

        FLOAT m_lastMillis;

        PVD::PvdConnection* m_pDebugConnection;

        physx::PxActor* AddActor(physx::PxGeometry& geo, IActor* actor, CONST util::Vec3& offsetPosition, std::string& material, FLOAT density);

        px::Material& CheckMaterial(std::string material) {

            auto it = this->m_materials.find(material);

            if(it == m_materials.end())
            {
                LOG_CRITICAL_ERROR_A("Material: '%s' not found", material.c_str());
            }

            return it->second;
        }

        std::map<std::string, px::Material> m_materials;

        VOID CreateTriangleConcaveMesh(IActor* actor, CONST IMesh* mesh, CONST util::Vec3& position, std::string& material);
        VOID CreateTriangleConvexMesh(IActor* actor, CONST IMesh* mesh, CONST util::Vec3& position, std::string& material);

        VOID CreateFromActor(IActor* actor);

    public:

        PhysX(VOID);

        BOOL VInit(VOID);

        VOID VCreateStaticPlane(CONST util::Vec3& dimension, IActor* actor, std::string& material);

        VOID VCreateSphere(FLOAT radius, IActor* actor, CONST util::Vec3& offsetPosition, std::string& material);

        VOID VCreateCube(CONST util::Vec3& dimension, IActor* actor, CONST util::Vec3& offsetPosition, std::string& material);

        VOID VCreateTriangleMesh(IActor* actor, CONST IMesh* mesh, CONST util::Vec3& position, std::string& material, std::string& shapeType);

        VOID VCreateTrigger(FLOAT radius, IActor* actor);

        VOID VRemoveActor(ActorId id);

        VOID VApplyForce(CONST util::Vec3& dir, FLOAT newtons, IActor* actor);

        VOID VApplyTorque(CONST util::Vec3& dir, FLOAT newtons, IActor* actor);

        VOID VCreateCharacterController(ActorId id, CONST util::Vec3& pos, FLOAT radius, FLOAT height);

        VOID VMoveKinematic(IActor* actor, CONST util::Vec3* pos, CONST util::Vec4* quat, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE);

        VOID VMoveKinematic(IActor* actor, CONST util::Vec3* pos, CONST util::Vec3* axis, FLOAT angle, FLOAT deltaMillis, BOOL isDeltaMove, BOOL isJump = FALSE);

        VOID VDebugRender(VOID);

        VOID VSyncScene(VOID);

        VOID VUpdate(FLOAT deltaMillis);

        VOID NewComponentDelegate(IEventPtr data);

        VOID ApplyForceTorqueDelegate(IEventPtr data);

        VOID OnResourceChanged(IEventPtr data);

        ~PhysX(VOID);
    };
};