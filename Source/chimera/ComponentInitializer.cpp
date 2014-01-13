#include "ActorFactory.h"
#include "Components.h"
#include "tinyxml2.h"
#include "Camera.h"
#include "GameView.h"

namespace chimera
{
    BOOL InitiaizeTransformComponent(IActorComponent* cmp, ICMStream* stream)
    {
        TransformComponent* transformation = (TransformComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;
        tinyxml2::XMLElement* trans = pData->FirstChildElement("Position");

        if(trans)
        {
            FLOAT x, y, z;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != trans->QueryFloatAttribute("z", &z));
            transformation->GetTransformation()->Translate(x, y, z);
        }

        tinyxml2::XMLElement* rot = pData->FirstChildElement("Rotation");

        if(rot)
        {
            FLOAT x, y, z, w;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("z", &z));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != rot->QueryFloatAttribute("w", &w));
            transformation->GetTransformation()->SetRotateQuat(x, y, z, w);
        }

        tinyxml2::XMLElement* scale = pData->FirstChildElement("Scale");

        if(scale)
        {
            FLOAT x, y, z;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("x", &x));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("y", &y));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != scale->QueryFloatAttribute("z", &z));
            transformation->GetTransformation()->SetScale(x, y, z);
        }
        
        return TRUE;
    }

    BOOL InitiaizeRenderingComponent(IActorComponent* cmp, ICMStream* stream)
    {
        RenderComponent* rendering = (RenderComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;

        tinyxml2::XMLElement* source = pData->FirstChildElement("Resource");

        if(source)
        {
            rendering->m_resource = source->GetText();
        }

        tinyxml2::XMLElement* type = pData->FirstChildElement("Type");

        if(type)
        {
            rendering->m_type = type->GetText();
        }

        tinyxml2::XMLElement* info = pData->FirstChildElement("Info");

        if(info)
        {
            //rendering->m_info = info->GetText();
        }

        return TRUE;
    }

    BOOL InitiaizeCameraComponent(IActorComponent* cmp, ICMStream* stream)
    {
        CameraComponent* cameraCmp = (CameraComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;

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
                cameraCmp->m_type = type->GetText();
            }
            else
            {
                cameraCmp->m_type = std::string("FreeLook");
            }


            if(cameraCmp->m_type == std::string("FPSCharacter"))
            {
                cameraCmp->m_camera = std::shared_ptr<util::FPSCamera>(new util::CharacterCamera(
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), 
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    );
            }
            else if(cameraCmp->m_type == std::string("FPSShakeCharacter"))
            {
                cameraCmp->m_camera = std::shared_ptr<util::FPSCamera>(new util::CharacterHeadShake(
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), 
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); 
            }
            else if(cameraCmp->m_type == "FPSStatic")
            {
                cameraCmp->m_camera = std::shared_ptr<util::StaticCamera>(new util::StaticCamera(
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), 
                    chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight(), 
                    (FLOAT)n, 
                    (FLOAT)f)
                    ); 
            }
            else
            {
                LOG_CRITICAL_ERROR("FPS Camera is bugged"); //use FPSCharacter

            }
            cameraCmp->m_camera->SetFoV(DEGREE_TO_RAD(fov));

            tinyxml2::XMLElement* activate = pData->FirstChildElement("Activate");

            if(activate)
            {
                CmGetApp()->VGetHumanView()->VSetTarget(cameraCmp->VGetActor());
            }

            return TRUE;
        }

        return FALSE;
    }

    BOOL InitiaizeControllerComponent(IActorComponent* cmp, ICMStream* stream)
    {
        ControllerComponent* controllerCmp = (ControllerComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;

        CharacterController* cc = new CharacterController();
        cc->SetName(cmp->VGetActor()->GetName());

        tinyxml2::XMLElement* speed = pData->FirstChildElement("Speed");
        if(speed)
        {
            FLOAT mins = 1;
            FLOAT maxs = 1;
            speed->QueryFloatAttribute("min", &mins);
            speed->QueryFloatAttribute("max", &maxs);
            cc->VSetMinSpeed(mins);
            cc->VSetMaxSpeed(maxs);
        }
        std::unique_ptr<ActorController> ac(cc);

        tinyxml2::XMLElement* activate = pData->FirstChildElement("Activate");
        if(activate)
        {
            cc->VActivate();
        }

        CmGetApp()->VGetLogic()->VAttachView(std::move(ac), cmp->VGetActor());

        return TRUE;
    }

    BOOL InitiaizePhysicsComponent(IActorComponent* cmp, ICMStream* stream)
    {
        PhysicComponent* phxCmp = (PhysicComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;

        tinyxml2::XMLElement* shapeStyle = pData->FirstChildElement("ShapeStyle");

        if(shapeStyle)
        {
            phxCmp->m_shapeStyle = shapeStyle->GetText();
        }

        tinyxml2::XMLElement* shapeType = pData->FirstChildElement("ShapeType");

        if(shapeType)
        {
            phxCmp->m_shapeType = shapeType->GetText();
        }

        tinyxml2::XMLElement* meshFile = pData->FirstChildElement("MeshFile");

        if(meshFile)
        {
            phxCmp->m_meshFile = meshFile->GetText();
        }

        tinyxml2::XMLElement* material = pData->FirstChildElement("Material");

        if(material)
        {
            phxCmp->m_material = material->GetText();
        }


        return TRUE;
    }

    BOOL InitiaizeLightComponent(IActorComponent* cmp, ICMStream* stream)
    {
        LightComponent* lightComp = (LightComponent*)cmp;
        tinyxml2::XMLElement* pData = (tinyxml2::XMLElement*)stream;

        tinyxml2::XMLElement* type = pData->FirstChildElement("Type");
        if(type)
        {
            lightComp->m_type = type->GetText();
        }

        tinyxml2::XMLElement* color = pData->FirstChildElement("Color");
        if(color)
        {
            FLOAT r, g, b;
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != color->QueryFloatAttribute("r", &r));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != color->QueryFloatAttribute("g", &g));
            RETURN_IF_FAILED(tinyxml2::XML_NO_ATTRIBUTE != color->QueryFloatAttribute("b", &b));
            lightComp->m_color.x = r;
            lightComp->m_color.y = g;
            lightComp->m_color.z = b;
        }

        tinyxml2::XMLElement* intensity = pData->FirstChildElement("Intensity");
        if(intensity)
        {
            lightComp->m_intensity = atof(intensity->GetText());
        }

        tinyxml2::XMLElement* angle = pData->FirstChildElement("Angle");
        if(angle)
        {
            lightComp->m_angle = atof(angle->GetText());
        }

        tinyxml2::XMLElement* texture = pData->FirstChildElement("Texture");
        if(texture)
        {
            lightComp->m_projTexture = texture->GetText();
        }

        tinyxml2::XMLElement* activate = pData->FirstChildElement("Activate");
        lightComp->m_activated = activate != NULL;

        tinyxml2::XMLElement* castShadow = pData->FirstChildElement("CastShadow");
        if(castShadow)
        {
            lightComp->m_castShadow = atof(castShadow->GetText()) > 0;
        }

        return TRUE;
    }
}