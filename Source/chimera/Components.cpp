#include "Components.h"
#include "Vec3.h"
#include "EventManager.h"
#include "GameApp.h"
#include "Thread.h"
#include "math.h"
#include "tinyxml2.h"
#include "GameView.h"
#include "D3DRenderer.h"
#include "Camera.h"

namespace tbd 
{

    CONST ComponentId TransformComponent::COMPONENT_ID = 0xdb756713;
    CONST ComponentId RenderComponent::COMPONENT_ID = 0x8beb1acc;
    CONST ComponentId CameraComponent::COMPONENT_ID = 0xb8a716ca;
    CONST ComponentId PhysicComponent::COMPONENT_ID = 0xc1514f;
    CONST ComponentId LightComponent::COMPONENT_ID = 0x1b5b0ea4;
    CONST ComponentId PickableComponent::COMPONENT_ID = 0xd295188c;
    CONST ComponentId ParticleComponent::COMPONENT_ID = 0x746a7b4a;
    CONST ComponentId SoundEmitterComponent::COMPONENT_ID = 0x568a0c05;

    ActorComponent::ActorComponent(VOID) : m_waitTillHandled(FALSE), m_handle(NULL)
    {
        m_handle = CreateEvent(NULL, FALSE, FALSE, NULL);
    }

    VOID ActorComponent::VPostInit(VOID)
    {
        std::shared_ptr<tbd::Actor> owner = this->m_owner.lock();
        event::IEventPtr event = std::shared_ptr<event::NewComponentCreatedEvent>(new event::NewComponentCreatedEvent(GetComponentId(), owner->GetId()));
        event::IEventManager::Get()->VQueueEventThreadSave(event);
    }

    VOID ActorComponent::VSetHandled(VOID)
    {
        if(m_handle)
        {
            SetEvent(m_handle);
        }
    }

    ActorComponent::~ActorComponent(VOID)
    {
        if(m_handle)
        {
            CloseHandle(m_handle);
        }
    }

    BOOL ParticleComponent::VInit(tinyxml2::XMLElement* pData)
    {

        return TRUE;
    }

    VOID ParticleComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {

    }

    BOOL TransformComponent::VInit(tinyxml2::XMLElement* pData) 
    {
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
            FLOAT x, y, z;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("z", &z));
            m_transformation.SetRotation((FLOAT)DEGREE_TO_RAD(x), (FLOAT)DEGREE_TO_RAD(y), (FLOAT)DEGREE_TO_RAD(z));
        }

        return TRUE;
    }

    VOID TransformComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {

        tinyxml2::XMLElement* transform = pData->NewElement("TransformComponent");
        tinyxml2::XMLElement* position = pData->NewElement("Position");

        position->SetAttribute("x", m_transformation.GetTranslation().x);
        position->SetAttribute("y", m_transformation.GetTranslation().y);
        position->SetAttribute("z", m_transformation.GetTranslation().z);

        tinyxml2::XMLElement* rotation = pData->NewElement("Rotation");
        util::Vec3 pyr;
        m_transformation.GetPitchYawRoll(pyr);
        rotation->SetAttribute("x", RAD_TO_DEGREE(pyr.x));
        rotation->SetAttribute("y", RAD_TO_DEGREE(pyr.y));
        rotation->SetAttribute("z", RAD_TO_DEGREE(pyr.z));

        transform->LinkEndChild(position);

        transform->LinkEndChild(rotation);

        pData->LinkEndChild(transform);
    }

    BOOL CameraComponent::VInit(tinyxml2::XMLElement* pData) 
    {
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
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    );
            }
            else if(m_type == "FPSShakeCharacter")
            {
                m_camera = std::shared_ptr<util::FPSCamera>(new util::CharacterHeadShake(
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    );
            }
            else if(m_type == "FPSStatic")
            {
                m_camera = std::shared_ptr<util::StaticCamera>(new util::StaticCamera(
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    );
            }
            else
            {
                LOG_CRITICAL_ERROR("FPS Camera is bugged"); //use FPSCharacter
                /*m_camera = std::shared_ptr<util::FPSCamera>(new util::FPSCamera(
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetWidth(), 
                    app::g_pApp->GetHumanView()->GetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); */
            }
            //RETURN_IF_FAILED(type);

            m_camera->SetFoV(DEGREE_TO_RAD(fov));
            /*this->m_aspect = (FLOAT)aspect;
            this->m_far = (FLOAT)f;
            this->m_near = (FLOAT)n;
            this->m_FoV = (FLOAT)fov / 180.0f * XM_PI;
            this->m_viewDir.Set((INT)viewDir == 0, (INT)viewDir == 1, (INT)viewDir == 2); */

            return TRUE;
        }
        else
        {
            return FALSE;
        }
    }

    RenderComponent::RenderComponent(VOID)
    {
        m_type = "mesh";
        m_anchorType = "sphere";
        m_drawType = "solid";
        m_anchorRadius = 1.0f;
        m_anchorBoxHE.Set(1, 1, 1);
        m_sceneNode = NULL;
        WaitTillHandled();
    }

    BOOL RenderComponent::VInit(tinyxml2::XMLElement* pData) 
    {
        tinyxml2::XMLElement* source = pData->FirstChildElement("MeshFile");
        //RETURN_IF_FAILED(source);
        if(source)
        {
            m_meshFile = tbd::Resource(source->GetText());
        }

        tinyxml2::XMLElement* type = pData->FirstChildElement("Type");

        if(type)
        {
            m_type = type->GetText();
        }

        return TRUE;
    }

    VOID RenderComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {
        tinyxml2::XMLElement* cmp = pData->NewElement("RenderComponent");
        tinyxml2::XMLElement* source = pData->NewElement("MeshFile");
        tinyxml2::XMLText* text = pData->NewText(m_meshFile.m_name.c_str());
        cmp->LinkEndChild(source);
        source->LinkEndChild(text);
        pData->LinkEndChild(cmp);
    }

    VOID RenderComponent::VCreateResources(VOID)
    {
        if(m_meshFile.m_name != "unknown")
        {
            app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_meshFile);
        }
        //DEBUG_OUT(m_meshFile.m_name + " done");
    }

    BOOL PhysicComponent::VInit(tinyxml2::XMLElement* pData) 
    {
         tinyxml2::XMLElement* shape = pData->FirstChildElement("Shape");
         RETURN_IF_FAILED(shape);

         m_shapeType = shape->GetText();

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
             m_meshFile = tbd::Resource(file->GetText());
         }
 
         return TRUE;
    }

    VOID PhysicComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {
        tinyxml2::XMLElement* cmp = pData->NewElement("PhysicComponent");

        tinyxml2::XMLElement* elem = pData->NewElement("Material");
        tinyxml2::XMLText* text = pData->NewText(m_material.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = pData->NewElement("Shape");
        text = pData->NewText(m_shapeType.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = pData->NewElement("MeshFile");
        text = pData->NewText(m_meshFile.m_name.c_str());
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = pData->NewElement("Dimension");

        elem->SetAttribute("x", m_dim.x);
        elem->SetAttribute("y", m_dim.y);
        elem->SetAttribute("z", m_dim.z);
        cmp->LinkEndChild(elem);
        pData->LinkEndChild(cmp);
    }

    VOID PhysicComponent::VCreateResources(VOID)
    {
        tbd::Resource r;
        if(m_meshFile.m_name != r.m_name)
        {
            app::g_pApp->GetCache()->GetHandle(m_meshFile);
        }
    }

    BOOL LightComponent::VInit(tinyxml2::XMLElement* pData)
    {
        m_activated = TRUE;

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

        /*tinyxml2::XMLElement* rad = pData->FirstChildElement("Radius");
        RETURN_IF_FAILED(rad);
        DOUBLE radius = atof(rad->GetText());
        if(radius == 0.0) return FALSE;
        m_radius = (FLOAT)radius; */
        return TRUE;
    }

    VOID LightComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {
        
        tinyxml2::XMLElement* cmp = pData->NewElement("LightComponent");
        //cmp->
        tinyxml2::XMLElement* elem = pData->NewElement("Type");
        tinyxml2::XMLText* text = pData->NewText(m_type.c_str());

        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);

        elem = pData->NewElement("Color");
        elem->SetAttribute("r", m_color.x * 255);
        elem->SetAttribute("g", m_color.y * 255);
        elem->SetAttribute("b", m_color.z * 255);
        cmp->LinkEndChild(elem);

        /*elem = new tinyxml2::XMLElement("Radius");
        std::stringstream ss;
        ss << m_radius;
        text = new TiXmlText(ss.str().c_str()); */
        
        elem->LinkEndChild(text);
        cmp->LinkEndChild(elem);
        pData->LinkEndChild(cmp);
    }

    VOID PickableComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {
        tinyxml2::XMLElement* elem = pData->NewElement("PickableComponent");
        pData->LinkEndChild(elem);
    }

    SoundEmitterComponent::SoundEmitterComponent(VOID) : m_loop(FALSE), m_soundFile("Unknown")
    {
        WaitTillHandled();
    }

    BOOL SoundEmitterComponent::VInit(tinyxml2::XMLElement* pData)
    {
        tinyxml2::XMLElement* file = pData->FirstChildElement("SoundFile");
        RETURN_IF_FAILED(file);
        m_soundFile = file->GetText();

        tinyxml2::XMLElement* loop = pData->FirstChildElement("Loop");
        RETURN_IF_FAILED(loop);
        m_loop = (file->GetText() == "true") || (file->GetText() == "TRUE");

        tinyxml2::XMLElement* radius = pData->FirstChildElement("Radius");
        RETURN_IF_FAILED(radius);

        m_radius = (FLOAT)atof(radius->GetText());

        if(m_radius == 0)
        {
            LOG_CRITICAL_ERROR("radius 0");
        }

        return TRUE;
    }

    VOID SoundEmitterComponent::VSave(tinyxml2::XMLDocument* pData) CONST
    {

    }

    VOID SoundEmitterComponent::VCreateResources(VOID)
    {
        //warum up cache
        tbd::Resource r(m_soundFile);
        app::g_pApp->GetCache()->GetHandle(r);
    }
}
