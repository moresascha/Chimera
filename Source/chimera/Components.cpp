#include "Components.h"
#include "tinyxml2.h"
#include "Event.h"

namespace chimera 
{
    ActorComponent::ActorComponent(VOID)
    {

    }

    VOID ActorComponent::VPostInit(VOID)
    {

    }

    ActorComponent::~ActorComponent(VOID)
    {
    }

    TransformComponent::TransformComponent(VOID) : m_phi(0), m_theta(0)
    {

    }

    BOOL TransformComponent::VInitialize(IStream* stream)
    {
        /*
        tinyxml2::XMLElement* trans = pData->FirstChildElement("Position");

        if(trans)
        {
            FLOAT x, y, z;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("z", &z));
            m_transformation.Translate(x, y, z);
        }

        tinyxml2::XMLElement* rot = pData->FirstChildElement("Rotation");

        if(rot)
        {
            FLOAT x, y, z, w;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("z", &z));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("w", &w));
            m_transformation.SetRotateQuat(x, y, z, w);
        }

        tinyxml2::XMLElement* scale = pData->FirstChildElement("Scale");

        if(scale)
        {
            FLOAT x, y, z;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("z", &z));
            m_transformation.SetScale(x, y, z);
        }
        */
        return TRUE;
    }

    VOID TransformComponent::VSerialize(IStream* stream) CONST
    {
        /*
        tinyxml2::XMLDocument* doc = pData->GetDocument();
        tinyxml2::XMLElement* transform = doc->NewElement("TransformComponent");
        tinyxml2::XMLElement* position = doc->NewElement("Position");

        position->SetAttribute("x", m_transformation.GetTranslation().x);
        position->SetAttribute("y", m_transformation.GetTranslation().y);
        position->SetAttribute("z", m_transformation.GetTranslation().z);

        tinyxml2::XMLElement* rotation = doc->NewElement("Rotation");

        CONST util::Vec4& rot = m_transformation.GetRotation();

        rotation->SetAttribute("x", rot.x);
        rotation->SetAttribute("y", rot.y);
        rotation->SetAttribute("z", rot.z);
        rotation->SetAttribute("w", rot.w);

        tinyxml2::XMLElement* scale = doc->NewElement("Scale");
        scale->SetAttribute("x", m_transformation.GetScale().x);
        scale->SetAttribute("y", m_transformation.GetScale().y);
        scale->SetAttribute("z", m_transformation.GetScale().z);

        transform->LinkEndChild(position);

        transform->LinkEndChild(scale);

        transform->LinkEndChild(rotation);

        pData->LinkEndChild(transform); */
    }

    BOOL CameraComponent::VInitialize(IStream* stream)
    {
        /*
        tinyxml2::XMLElement* settings = pData->FirstChildElement("Settings");
        if(settings)
        {
            FLOAT fov, aspect, viewDir, n, f;
        
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != settings->QueryFloatAttribute("fov", &fov));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != settings->QueryFloatAttribute("aspect", &aspect));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != settings->QueryFloatAttribute("viewAxis", &viewDir));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != settings->QueryFloatAttribute("far", &f));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != settings->QueryFloatAttribute("near", &n));
        
            //TODO: do we have to keep it?
            tinyxml2::XMLElement* type = pData->FirstChildElement("Type");
            if(type)
            {
                m_type = type->GetText();
            }
            else
            {
                m_type = std::string("FreeLook");
            }


            if(m_type == "FPSCharacter")
            {
                m_camera = std::shared_ptr<util::FPSCamera>(new util::CharacterCamera(
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    );
            }
            else if(m_type == "FPSShakeCharacter")
            {
                m_camera = std::shared_ptr<util::FPSCamera>(new util::CharacterHeadShake(
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); 
            }
            else if(m_type == "FPSStatic")
            {
                m_camera = std::shared_ptr<util::StaticCamera>(new util::StaticCamera(
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    chimera::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); 
            }
            else
            {
                LOG_CRITICAL_ERROR("FPS Camera is bugged"); //use FPSCharacter
                m_camera = std::shared_ptr<util::FPSCamera>(new util::FPSCamera(
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); 
            }
            //RETURN_IF_FAILED(type);

            m_camera->SetFoV(DEGREE_TO_RAD(fov));
            this->m_aspect = (FLOAT)aspect;
            this->m_far = (FLOAT)f;
            this->m_near = (FLOAT)n;
            this->m_FoV = (FLOAT)fov / 180.0f * XM_PI;
            this->m_viewDir.Set((INT)viewDir == 0, (INT)viewDir == 1, (INT)viewDir == 2);

            return TRUE;
        }
        else
        {
            return FALSE;
        }
        */
        return TRUE;
    }

    RenderComponent::RenderComponent(VOID)
    {
        m_type = "mesh";
        m_anchorType = "sphere";
        m_drawType = "solid";
        m_anchorRadius = 1.0f;
        m_anchorBoxHE.Set(1, 1, 1);
        m_sceneNode = NULL;
    }

    BOOL RenderComponent::VInitialize(IStream* stream)
    {
        /*
        tinyxml2::XMLElement* source = pData->FirstChildElement("MeshFile");
        //RETURN_IF_FAILED(source);
        if(source)
        {
            m_meshFile = CMResource(source->GetText());
        }

        tinyxml2::XMLElement* type = pData->FirstChildElement("Type");

        if(type->GetText())
        {
            m_type = type->GetText();
        }

        tinyxml2::XMLElement* info = pData->FirstChildElement("Info");

        if(info->GetText())
        {
            m_info = info->GetText();
        }
        */
        return TRUE;
    }

    VOID RenderComponent::VSerialize(IStream* stream) CONST
    {
        /*
        tinyxml2::XMLDocument* doc = pData->GetDocument();
        tinyxml2::XMLElement* cmp = doc->NewElement("RenderComponent");
        tinyxml2::XMLElement* source = doc->NewElement("MeshFile");
        tinyxml2::XMLText* text = doc->NewText(m_meshFile.m_name.c_str());

        cmp->LinkEndChild(source);
        source->LinkEndChild(text);

        source = doc->NewElement("Type");
        text = doc->NewText(m_type.c_str());

        cmp->LinkEndChild(source);
        source->LinkEndChild(text);

        source = doc->NewElement("Info");
        text = doc->NewText(m_info.c_str());

        cmp->LinkEndChild(source);
        source->LinkEndChild(text);

        pData->LinkEndChild(cmp); */
    }

    VOID RenderComponent::VCreateResources(VOID)
    {
        CMResource c;
        if(m_resource != c)
        {
            CmGetApp()->VGetHumanView()->VGetVRamManager()->VGetHandle(CMResource(m_resource));
        }
/*
        if(CMResource(m_info) != c)
        {
            CmGetApp()->VGetCache()->VGetHandle(CMResource(m_info));
        }*/
    }

    BOOL PhysicComponent::VInitialize(IStream* stream)
    {
        /*
         tinyxml2::XMLElement* shape = pData->FirstChildElement("Shape");
         RETURN_IF_FAILED(shape);

         m_shapeStyle = shape->GetText();

         if(shape->Attribute("type"))
         {
             m_shapeType = shape->Attribute("type");
         }

         tinyxml2::XMLElement* dim = pData->FirstChildElement("Dimension");

         if(dim)
         {
             m_dim.x = dim->FloatAttribute("x");
             m_dim.y = dim->FloatAttribute("y");
             m_dim.z = dim->FloatAttribute("z");
             m_radius = dim->FloatAttribute("radius");
         }

         tinyxml2::XMLElement* material = pData->FirstChildElement("Material");

         if(material)
         {
            m_material = material->GetText();
         }
         else
         {
            m_material = "default";
         }

         tinyxml2::XMLElement* file = pData->FirstChildElement("MeshFile");

         if(file)
         {
             m_meshFile = chimera::CMResource(file->GetText());
         }
         */
         return TRUE;
    }

    VOID PhysicComponent::VSerialize(IStream* stream) CONST
    {
        /*
        tinyxml2::XMLDocument* doc = pData->GetDocument();

        tinyxml2::XMLElement* cmp = doc->NewElement("PhysicComponent");

        tinyxml2::XMLElement* elem = doc->NewElement("Material");
        tinyxml2::XMLText* text = doc->NewText(m_material.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("Shape");
        elem->SetAttribute("type", m_shapeType.c_str());
        text = doc->NewText(m_shapeStyle.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("MeshFile");
        text = doc->NewText(m_meshFile.m_name.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("Dimension");

        elem->SetAttribute("x", m_dim.x);
        elem->SetAttribute("y", m_dim.y);
        elem->SetAttribute("z", m_dim.z);
        elem->SetAttribute("radius", m_radius);
        cmp->LinkEndChild(elem);
        pData->LinkEndChild(cmp); */
    }

    VOID PhysicComponent::VCreateResources(VOID)
    {
        CMResource c;
        if(m_meshFile != c)
        {
            CmGetApp()->VGetCache()->VGetHandle(CMResource(m_meshFile));
        }
    }

    BOOL LightComponent::VInitialize(IStream* stream)
    {
        m_activated = TRUE;
        /*
        tinyxml2::XMLElement* file = pData->FirstChildElement("Type");
        RETURN_IF_FAILED(file);
        m_type = file->GetText();

        tinyxml2::XMLElement* color = pData->FirstChildElement("Color");
        RETURN_IF_FAILED(color);
        FLOAT r, g, b;
        r = color->FloatAttribute("r"); 
        g = color->FloatAttribute("g"); 
        b = color->FloatAttribute("b"); 
        m_color.x = (FLOAT) r / 255.0f;
        m_color.y = (FLOAT) g / 255.0f;
        m_color.z = (FLOAT) b / 255.0f;
        m_color.w = 1;

        tinyxml2::XMLElement* a = pData->FirstChildElement("Angle");
        m_angle = (FLOAT)atof(a->GetText());

        a = pData->FirstChildElement("Intensity");
        m_intensity = (FLOAT)atof(a->GetText());
        */
        return TRUE;
    }

    VOID LightComponent::VSerialize(IStream* stream) CONST
    {
        /*
        tinyxml2::XMLDocument* doc = pData->GetDocument();
        tinyxml2::XMLElement* cmp = doc->NewElement("LightComponent");
        //cmp->
        tinyxml2::XMLElement* elem = doc->NewElement("Type");
        tinyxml2::XMLText* text = doc->NewText(m_type.c_str());

        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("Color");
        elem->SetAttribute("r", m_color.x * 255);
        elem->SetAttribute("g", m_color.y * 255);
        elem->SetAttribute("b", m_color.z * 255);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("Angle");
        std::stringstream ss;
        ss << m_angle;
        text = doc->NewText(ss.str().c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = doc->NewElement("Intensity");
        ss.str("");
        ss << m_intensity;
        text = doc->NewText(ss.str().c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        pData->LinkEndChild(cmp);*/
    }

    VOID PickableComponent::VSerialize(IStream* stream) CONST
    {
        /*tinyxml2::XMLDocument* doc = pData->GetDocument();
        tinyxml2::XMLElement* elem = doc->NewElement("PickableComponent");
        pData->LinkEndChild(elem); */
    }

    SoundComponent::SoundComponent(VOID) : m_loop(FALSE), m_soundFile("Unknown"), m_emitter(FALSE)
    {

    }

    VOID SoundComponent::VCreateResources(VOID)
    {
        CmGetApp()->VGetCache()->VGetHandle(CMResource(m_soundFile));
    }

    BOOL SoundComponent::VInitialize(IStream* stream)
    {
        /*
        tinyxml2::XMLElement* file = pData->FirstChildElement("SoundFile");
        RETURN_IF_FAILED(file);
        m_soundFile = file->GetText();

        tinyxml2::XMLElement* loop = pData->FirstChildElement("Loop");
        RETURN_IF_FAILED(loop);
        m_loop = !strcmp(loop->GetText(), "true") || !strcmp(loop->GetText(), "TRUE");

        tinyxml2::XMLElement* radius = pData->FirstChildElement("Radius");
        RETURN_IF_FAILED(radius);

        m_radius = (FLOAT)atof(radius->GetText());

        if(m_radius == 0)
        {
            LOG_CRITICAL_ERROR("radius 0");
        }

        tinyxml2::XMLElement* emit = pData->FirstChildElement("Emitter");
        m_emitter = !strcmp(emit->GetText(), "true");*/

        return TRUE;
    }

    VOID SoundComponent::VSerialize(IStream* stream) CONST
    {

    }

    class SetParentWaitProcess : public IProcess
    {
    private:
        ActorId m_actor;
        ActorId m_parent;
        BOOL m_gotActor;
        BOOL m_gotParent;

        VOID EventListener(IEventPtr event)
        {
            if(event->VGetEventType() == CM_EVENT_ACTOR_CREATED)
            {
                std::shared_ptr<ActorCreatedEvent> e = std::static_pointer_cast<ActorCreatedEvent>(event);
                if(!m_gotActor)
                {
                    m_gotActor = e->m_id == m_actor;
                }
                if(!m_gotParent)
                {
                    m_gotParent = e->m_id == m_parent;
                }
            }
        }

    public:
        SetParentWaitProcess(ActorId actor, ActorId parent) : m_actor(actor), m_parent(parent), m_gotActor(FALSE), m_gotParent(FALSE)
        {

        }

        VOID VOnInit(VOID)
        {
            ADD_EVENT_LISTENER(this, &SetParentWaitProcess::EventListener, CM_EVENT_ACTOR_CREATED);
        }
        VOID VOnAbort(VOID)
        {
            REMOVE_EVENT_LISTENER(this, &SetParentWaitProcess::EventListener, CM_EVENT_ACTOR_CREATED);
        }
        VOID VOnFail(VOID)
        {
            REMOVE_EVENT_LISTENER(this, &SetParentWaitProcess::EventListener, CM_EVENT_ACTOR_CREATED);
        }
        VOID VOnSuccess(VOID)
        {
            REMOVE_EVENT_LISTENER(this, &SetParentWaitProcess::EventListener, CM_EVENT_ACTOR_CREATED);
        }

        VOID VOnUpdate(ULONG deltaMillis)
        {
            if(m_gotActor && m_gotParent || 
                (CmGetApp()->VGetHumanView()->VGetSceneGraph()->VFindActorNode(m_actor) && 
                CmGetApp()->VGetHumanView()->VGetSceneGraph()->VFindActorNode(m_parent)))
            {
                SetParentActorEvent* spe = new SetParentActorEvent();
                spe->m_actor = m_actor;
                spe->m_parentActor = m_parent;
                QUEUE_EVENT(spe);
                Succeed();
            }
        }
    };

    VOID ParentComponent::VPostInit(VOID)
    {
        CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<SetParentWaitProcess>(new SetParentWaitProcess(m_owner->GetId(), m_parentId)));
    }
}
