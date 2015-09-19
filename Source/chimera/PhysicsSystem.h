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

        virtual bool VInit(void) = 0;
    
        virtual void VCreateStaticPlane(const util::Vec3& dimension, IActor* actor, std::string& material) = 0;

        virtual void VCreateSphere(float radius, IActor* actor, const util::Vec3& offsetPosition, std::string& material) = 0;

        virtual void VCreateCube(const util::Vec3& dimension, IActor* actor, const util::Vec3& offsetPosition, std::string& material) = 0;

        virtual void VCreateTrigger(float radius, IActor* actor) = 0;

        virtual void VCreateTriangleMesh(IActor* actor, const IMesh* mesh, const util::Vec3& offsetPosition, std::string& material, std::string & shapeType) = 0;

        virtual void VCreateCharacterController(ActorId id, const util::Vec3& pos, float radius, float height) = 0;

        virtual void VRemoveActor(ActorId id) = 0;

        virtual void VApplyForce(const util::Vec3& dir, float newtons, IActor* actor) = 0;

        virtual void VApplyTorque(const util::Vec3& dir, float newtons, IActor* actor) = 0;

        virtual void VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec4* quat, float deltaMillis, bool isDeltaMove, bool isJump = false) = 0;

        virtual void VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec3* axis, float angle, float deltaMillis, bool isDeltaMove, bool isJump = false) = 0;

        virtual void VDebugRender(void) { }

        virtual void VSyncScene(void) = 0;

        virtual ActorId VRayCast(const util::Vec3* pos,  const util::Vec3* dir) = 0;

        virtual void VUpdate(float deltaMillis) = 0;

        virtual ~IPhysicsSystem(void) {}
    };

    class IPhysicsDebugRenderer 
    {
    public:
        virtual void VDrawLine(const util::Vec3& start, const util::Vec3& end) = 0;
        virtual void VDrawCube(const util::Vec3& extends, const util::Vec3& position) = 0;
    };

    namespace px
    {
        struct Material 
        {
            float m_staticFriction;
            float m_dynamicFriction;
            float m_restitution;
            float m_mass;
            float m_angulardamping;
            //FLOAT m_
            physx::PxMaterial* m_material;
            Material(float staticFriction, float dynamicFriction, float restitution, float mass, float angularDamping, physx::PxPhysics* physx);

            Material(void) : m_material(NULL) {}

            ~Material(void) { }
        };
    }


    class PhysX : public IPhysicsSystem 
    {
        friend class DefaultFilterCallback;

        class Controller_
        {
        private:
            float m_jumpDy;
            float m_time;
            float m_duration;
            float m_p0;
            float m_p1;
            float m_maxJumpvalueFunc;
            float m_jumpVelo;
            float m_a;

        public:
            Controller_(void);
            void SetJumpSettings(float duration);
            void Update(float deltaMillis);
            void Move(float dx, float dy, float deltaMilli);
            void Jump(float dy);
            bool IsOnGround(void);
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

        float m_lastMillis;

        PVD::PvdConnection* m_pDebugConnection;

        physx::PxActor* AddActor(physx::PxGeometry& geo, IActor* actor, const util::Vec3& offsetPosition, std::string& material, float density);

        px::Material& CheckMaterial(std::string material) {

            auto it = this->m_materials.find(material);

            if(it == m_materials.end())
            {
                LOG_CRITICAL_ERROR_A("Material: '%s' not found", material.c_str());
            }

            return it->second;
        }

        std::map<std::string, px::Material> m_materials;

        void CreateTriangleConcaveMesh(IActor* actor, const IMesh* mesh, const util::Vec3& position, std::string& material);
        void CreateTriangleConvexMesh(IActor* actor, const IMesh* mesh, const util::Vec3& position, std::string& material);

        void CreateFromActor(IActor* actor);

    public:

        PhysX(void);

        bool VInit(void);

        void VCreateStaticPlane(const util::Vec3& dimension, IActor* actor, std::string& material);

        void VCreateSphere(float radius, IActor* actor, const util::Vec3& offsetPosition, std::string& material);

        void VCreateCube(const util::Vec3& dimension, IActor* actor, const util::Vec3& offsetPosition, std::string& material);

        void VCreateTriangleMesh(IActor* actor, const IMesh* mesh, const util::Vec3& position, std::string& material, std::string& shapeType);

        void VCreateTrigger(float radius, IActor* actor);

        void VRemoveActor(ActorId id);

        void VApplyForce(const util::Vec3& dir, float newtons, IActor* actor);

        void VApplyTorque(const util::Vec3& dir, float newtons, IActor* actor);

        void VCreateCharacterController(ActorId id, const util::Vec3& pos, float radius, float height);

        void VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec4* quat, float deltaMillis, bool isDeltaMove, bool isJump = false);

        void VMoveKinematic(IActor* actor, const util::Vec3* pos, const util::Vec3* axis, float angle, float deltaMillis, bool isDeltaMove, bool isJump = false);

        void VDebugRender(void);

        void VSyncScene(void);

        void VUpdate(float deltaMillis);

        void NewComponentDelegate(IEventPtr data);

        void DeleteComponentDelegate(IEventPtr data);

        void ApplyForceTorqueDelegate(IEventPtr data);

        ActorId PhysX::VRayCast(const util::Vec3* pos,  const util::Vec3* dir);

        void OnResourceChanged(IEventPtr data);

        ~PhysX(void);
    };
};