#pragma once
#include "stdafx.h"
#include "Mat4.h"

#define CM_CMP_TRANSFORM 0xdb756713
#define CM_CMP_RENDERING 0x8beb1acc
#define CM_CMP_CAMERA 0xb8a716ca
#define CM_CMP_PHX 0xc1514f
#define CM_CMP_LIGHT 0x1b5b0ea4
#define CM_CMP_PICKABLE 0xd295188c
#define CM_CMP_SOUND 0x568a0c05
#define CM_CMP_PARENT_ACTOR 0xde7b06f1
#define CM_CMP_CONTROLLER 0xc591361a
#define CM_CMP_PARTICLE 0x813462a7

#define CM_CREATE_CMP_HEADER(__type, __name) \
    ComponentId VGetComponentId(VOID) CONST { return __type; } \
    LPCSTR VGetName(VOID) CONST { return #__name; }

namespace chimera 
{
    class ActorComponent : public IActorComponent
    {
    protected:
        IActor* m_owner;
    public:
        ActorComponent(void);

        void VSetOwner(IActor* pOwner) { m_owner = pOwner; }

        virtual void VPostInit(void);

        virtual void VCreateResources(void) { }

        IActor* VGetActor(void) { return m_owner; }

        virtual ~ActorComponent(void);
    };

    class ControllerComponent : public ActorComponent
    {
    public:
        CM_CREATE_CMP_HEADER(CM_CMP_CONTROLLER, ControllerComponent);
    };

    class TransformComponent : public ActorComponent
    {
    private:
        util::Mat4 m_transformation;
    public:
        float m_phi;
        float m_theta;

        TransformComponent(void);

        util::Mat4* GetTransformation(void) 
        {
            return &m_transformation;
        }

        CM_CREATE_CMP_HEADER(CM_CMP_TRANSFORM, TransformComponent);
    };

    class RenderComponent : public ActorComponent 
    {
    public:
        std::list<util::Vec3> m_instances;

        std::string m_anchorType;

        std::string m_type;

        std::string m_drawType;

        std::string m_meshId;

        float m_anchorRadius;

        util::Vec3 m_anchorBoxHE;

        std::shared_ptr<chimera::ISceneNode> m_sceneNode;

        std::unique_ptr<IGeometry> m_geo;

        std::shared_ptr<IVertexBuffer> m_vmemInstances;

        RenderComponent(void);

        CMResource m_resource;

        void VCreateResources(void);

        CM_CREATE_CMP_HEADER(CM_CMP_RENDERING, RenderComponent);
    };

    class ParticleComponent : public ActorComponent
    {
    public:
        IParticleSystem* m_pSystem;
        std::vector<std::string> m_modifier;
        ParticleComponent(void) : m_pSystem(NULL) { }
        CM_CREATE_CMP_HEADER(CM_CMP_PARTICLE, ParticleComponent);
    };

    class CameraComponent : public ActorComponent
    {
    public:
        std::shared_ptr<ICamera> m_camera;
        std::string m_type;

        CameraComponent(void)
        {

        }
        std::shared_ptr<ICamera> GetCamera(void) { return m_camera; }

        void SetCamera(std::shared_ptr<ICamera> cam) { m_camera = cam; }

        CM_CREATE_CMP_HEADER(CM_CMP_CAMERA, CameraComponent);
    };

    class PhysicComponent : public ActorComponent 
    {

    public:
        std::string m_shapeStyle;
        std::string m_shapeType;
        std::string m_material;

        chimera::CMResource m_meshFile;
        std::string m_subMesh;

        float m_radius;
        util::Vec3 m_dim;

        PhysicComponent(void) : m_dim(1,1,1), m_radius(1)
        {

        }

        void VCreateResources(void);

        CM_CREATE_CMP_HEADER(CM_CMP_PHX, PhysicComponent); 
    };

    class LightComponent : public ActorComponent
    {

    public:
        std::string m_type;
        util::Vec4 m_color;
        float m_angle;
        float m_intensity;
        bool m_activated;
        std::string m_projTexture;
        bool m_castShadow;
        float m_radius;

        LightComponent(void) : m_angle(0), m_activated(true), m_intensity(1), m_color(1,1,1,1), m_projTexture("white.png"), m_castShadow(1), m_radius(20)
        {

        }

        CM_CREATE_CMP_HEADER(CM_CMP_LIGHT, LightComponent); 
    };

    class PickableComponent : public ActorComponent
    {
    public:
        bool VInit(tinyxml2::XMLElement* pData) { return true; }

        CM_CREATE_CMP_HEADER(CM_CMP_PICKABLE, PickableComponent); 
    };

    class SoundComponent : public ActorComponent
    {
    public:
        SoundComponent(void);
        std::string m_soundFile;
        float m_radius;
        bool m_emitter;
        bool m_loop;
        void VCreateResources(void);
        
        CM_CREATE_CMP_HEADER(CM_CMP_SOUND, SoundComponent); 
    };

    class ParentComponent : public ActorComponent
    {
    public:
        ActorId m_parentId;
    public:
        ActorId GetParent(void) { return m_parentId; }

        void VPostInit(void);

        CM_CREATE_CMP_HEADER(CM_CMP_PARENT_ACTOR, ParentComponent); 
    };
}