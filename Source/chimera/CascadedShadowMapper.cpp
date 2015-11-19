#include "CascadedShadowMapper.h"
#include "Frustum.h"
#include "Camera.h"
#include "Components.h"
#include "Event.h"

namespace chimera
{
//#define SHADOW_START_SIZE 1024

#define CSM_DEBUG

#ifndef _DEBUG
//#undef CSM_DEBUG
#endif

    struct _LightingBuffer
    {
        XMFLOAT4X4 m_view;
        XMFLOAT4X4 m_iView;
        XMFLOAT4X4 m_projection[3]; //TODO
        XMFLOAT4 m_lightPos;
        XMFLOAT4 m_intensity;
        XMFLOAT4 m_ambient;
        XMFLOAT4 m_distances;
    };

    CascadedShadowMapper::CascadedShadowMapper(UCHAR cascades) 
        : m_cascades(cascades), m_ppTargets(NULL), /*m_camera(1024, 1024, 0.01, 1000),*/ m_pCascadesSettings(NULL), m_ppBlurChain(NULL)
    {

        /*
#ifdef CSM_DEBUG
        for(USHORT i = 0; i < m_cascades; ++i)
        {
            m_cascadeCameraActor[i] = chimera::g_pApp->GetLogic()->VCreateActor("staticcamera.xml");
            std::stringstream ss;
            ss << "cascadeCam";
            ss << i;
            m_cascadeCameraActor[i]->SetName(ss.str());
        }
#endif
        */
        m_intensity = util::Vec3(1,1,1);

        m_ambient = util::Vec3(0.1f, 0.1f, 0.1f);

        std::unique_ptr<ActorDescription> desc = CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        CameraComponent* cc = desc->AddComponent<CameraComponent>(CM_CMP_CAMERA);
        cc->SetCamera(std::shared_ptr<ICamera>(new util::FPSCamera(1,1,1e-2f,1e3)));
        TransformComponent* tc = desc->AddComponent<TransformComponent>(CM_CMP_TRANSFORM);

        m_lightActorCamera = CmGetApp()->VGetLogic()->VCreateActor(std::move(desc));

        //util::Vec3 lightPos(1.0f,1.3f,0.6f);
        util::Vec3 lightPos(1.0f, 1, 0);
        lightPos.Normalize();
        lightPos.Scale(util::Vec3(1000.0f, 1000.0f, 1000.0f));
        tc->GetTransformation()->SetTranslation(lightPos.x, lightPos.y, lightPos.z);
        cc->GetCamera()->MoveToPosition(tc->GetTransformation()->GetTranslation());
        m_lightActorCamera->SetName("cascadeLightCamera");

        /*
        m_viewActor = chimera::g_pApp->GetLogic()->VCreateActor("rendercamera.xml");
        m_viewActor->SetName("cascadeViewCamera");
        m_viewActor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->SetTranslate(0,5,-5);
        m_viewActor->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock()->GetCamera()->MoveToPosition(
            m_viewActor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation());

        */
        ADD_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunPositionDelegate, CM_EVENT_SET_SUN_POSITION);
        ADD_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunIntensityDelegate, CM_EVENT_SET_SUN_INTENSITY);
        ADD_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunAmbientDelegate, CM_EVENT_SET_SUN_AMBIENT);
    }

    bool CascadedShadowMapper::VOnRestore(void)
    {
        uint startSize = CmGetApp()->VGetConfig()->VGetInteger("iCSMSize");

        if(!m_ppTargets)
        {
            m_ppTargets = new IRenderTarget*[m_cascades];
            m_ppBlurredTargets = new IRenderTarget*[m_cascades];
            m_ppBlurChain = new IEffectChain*[m_cascades];

            for(UCHAR i = 0; i < m_cascades; ++i)
            {
                m_ppTargets[i] = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget().release();
                m_ppBlurredTargets[i] = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateRenderTarget().release();

                uint dim = startSize / (1 << i);

                m_ppBlurChain[i] = CmGetApp()->VGetHumanView()->VGetEffectFactory()->VCreateEffectChain();

                CMShaderDescription desc;
                desc.file = L"Effects.hlsl";
                desc.function = "VSMBlurH";
                IEffect* e0 = m_ppBlurChain[i]->VAppendEffect(desc);
                e0->VAddSource(m_ppTargets[i]);

                desc.function = "VSMBlurV";
                IEffect* e1 = m_ppBlurChain[i]->VAppendEffect(desc);
                //e1->VAddRequirement(e0);
                e1->VSetTarget(m_ppBlurredTargets[i]);

                m_ppBlurChain[i]->VOnRestore(dim, dim);
            }
        }

        SAFE_ARRAY_DELETE(m_pCascadesSettings);

        m_pCascadesSettings = new CascadeSettings[m_cascades];

        float f = 1000;
        
        float factor = 3.5;
        float l = 0;
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            l += (float)pow(factor, i);
        }
        float t = 1.0f / l;
        float sum = 0;

        float lastZ = 0;

        for(UCHAR i = 1; i <= m_cascades; ++i)
        {
            uint dim = startSize / (1 << (i-1));
            m_ppTargets[i - 1]->VSetClearColor(1000, 1000, 1000, 1000);
            //m_ppTargets[i - 1]->SetMiscflags(D3D11_RESOURCE_MISC_GENERATE_MIPS);
            m_ppTargets[i - 1]->VOnRestore(dim, dim, eFormat_R32G32_FLOAT);

            m_pCascadesSettings[i - 1].start = -10;
            sum += pow(factor, (i-1)) * t;
            m_pCascadesSettings[i - 1].end = 14;

            m_ppBlurChain[i - 1]->VOnRestore(dim, dim);
        }

        float cuttOffset = 0;

        m_pCascadesSettings[0].start = 0;//-cuttOffset;
        m_pCascadesSettings[0].end = 16;

        m_pCascadesSettings[1].start = m_pCascadesSettings[0].end - cuttOffset;
        m_pCascadesSettings[1].end = 55;

        m_pCascadesSettings[2].start = m_pCascadesSettings[1].end - cuttOffset;
        m_pCascadesSettings[2].end = 500;

        /*m_cascadesSettings[3].start = 256;
        m_cascadesSettings[3].end = 1000; */
        CMShaderProgramDescription desc;

        desc.vs.file = L"CascadedShadowMap.hlsl";
        desc.vs.function = "CSM_VS";
        desc.vs.layoutCount = 3;
        desc.vs.inputLayout[0].name = "POSITION";
        desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;
        desc.vs.inputLayout[0].instanced = false;
        desc.vs.inputLayout[0].slot = 0;
        desc.vs.inputLayout[0].position = 0;

        desc.vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;
        desc.vs.inputLayout[1].name = "NORMAL";
        desc.vs.inputLayout[1].instanced = false;
        desc.vs.inputLayout[1].slot = 0;
        desc.vs.inputLayout[1].position = 1;

        desc.vs.inputLayout[2].name = "TEXCOORD";
        desc.vs.inputLayout[2].format = eFormat_R32G32_FLOAT;
        desc.vs.inputLayout[2].instanced = false;
        desc.vs.inputLayout[2].slot = 0;
        desc.vs.inputLayout[2].position = 2;

        desc.fs.file = desc.vs.file;
        desc.fs.function = "CSM_PS";

        m_pProgram = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("CSM", &desc);

        desc.vs.file = L"CascadedShadowMap.hlsl";
        desc.vs.function = "CSM_Instanced_VS";

        desc.vs.layoutCount = 4;

        desc.vs.inputLayout[0].instanced = false;
        desc.vs.inputLayout[0].name = "POSITION";
        desc.vs.inputLayout[0].position = 0;
        desc.vs.inputLayout[0].slot = 0;
        desc.vs.inputLayout[0].format = eFormat_R32G32B32_FLOAT;

        desc.vs.inputLayout[1].instanced = false;
        desc.vs.inputLayout[1].name = "NORMAL";
        desc.vs.inputLayout[1].position = 1;
        desc.vs.inputLayout[1].slot = 0;
        desc.vs.inputLayout[1].format = eFormat_R32G32B32_FLOAT;

        desc.vs.inputLayout[2].instanced = false;
        desc.vs.inputLayout[2].name = "TEXCOORD";
        desc.vs.inputLayout[2].position = 2;
        desc.vs.inputLayout[2].slot = 0;
        desc.vs.inputLayout[2].format = eFormat_R32G32_FLOAT;

        desc.vs.inputLayout[3].instanced = true;
        desc.vs.inputLayout[3].name = "TANGENT";
        desc.vs.inputLayout[3].position = 3;
        desc.vs.inputLayout[3].slot = 1;
        desc.vs.inputLayout[3].format = eFormat_R32G32B32_FLOAT;

        m_pProgramInstanced = CmGetApp()->VGetHumanView()->VGetRenderer()->VGetShaderCache()->VCreateShaderProgram("CSM_Instanced", &desc);

        util::Vec3 eyePos;
        eyePos = GetActorCompnent<TransformComponent>(m_lightActorCamera, CM_CMP_TRANSFORM)->GetTransformation()->GetTranslation();
        //m_camera.LookAt(eyePos, util::Vec3(0,0,0));

        CameraComponent* cc = GetActorCompnent<CameraComponent>(m_lightActorCamera, CM_CMP_CAMERA);
        cc->GetCamera()->LookAt(eyePos, util::Vec3(0,0,0));

        //m_camera.MoveToPosition(eyePos);
        //m_camera.Rotate(0, 0.785f);
        //m_camera.Rotate(0, XM_PIDIV2);
        //m_camera.GetView().Print();
        return true;
    }

    void CascadedShadowMapper::VRender(ISceneGraph* graph)
    {

        IRenderer* renderer = CmGetApp()->VGetHumanView()->VGetRenderer();
        IDeviceTexture* vn[4];
        for(UCHAR i = 0; i < m_cascades + 1; ++i)
        {
            vn[i] = NULL;
        }
        renderer->VSetTextures(eEffect0, (vn+1), m_cascades);

        ICamera* playerView = graph->VGetCamera().get();

        Frustum cascadeFrusta;
        Frustum ortographicFrusta;

        CameraComponent* lcc = GetActorCompnent<CameraComponent>(m_lightActorCamera, CM_CMP_CAMERA);

        //std::shared_ptr<chimera::CameraComponent> vcc = m_viewActor->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
        
        renderer->VPushViewTransform(lcc->GetCamera()->GetView(), lcc->GetCamera()->GetIView(), lcc->GetCamera()->GetEyePos(), lcc->GetCamera()->GetViewDir());

        ICamera* lightCamera = lcc->GetCamera().get();
        //util::ICamera* viewCamera = vcc->GetCamera().get();
        ICamera* viewCamera = playerView;

        //util::Mat4 viewToLight = util::Mat4::Mul(lightCamera->GetView(), viewCamera->GetIView());//vcc->GetCamera()->GetIView());

        float distances[3];

        for(UCHAR ci = 0; ci < m_cascades; ++ci)
        {
            CascadeSettings& settings = m_pCascadesSettings[ci];
            
            /*util::StaticCamera* staticCam;
            staticCam = (util::StaticCamera*)(m_cascadeCameraActor[ci]->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock()->GetCamera().get());*/

            float farr = settings.end;
            float nnear = settings.start;
 
            cascadeFrusta.CreatePerspective(viewCamera->GetAspect(), viewCamera->GetFoV(), nnear, farr);

            util::Vec3 vmin(FLT_MAX, FLT_MAX, FLT_MAX);
            util::Vec3 vmax(FLT_MIN, FLT_MIN, FLT_MIN);
            util::Vec3 vFrustumPoints[8];
            util::Vec3 vecs[8];

            for(uint i = 0; i < 8; ++i)
            {
                util::Vec3 v = util::Mat4::Transform(viewCamera->GetIView(), cascadeFrusta.GetPoints()[i]);

                vFrustumPoints[i] = v;

                util::Vec3 pos = util::Mat4::Transform(lightCamera->GetView(), v);

                vmax = util::Vec3::Max(vmax, pos);
                vmin = util::Vec3::Min(vmin, pos);
            }

            /*FLOAT bound = (farr - nnear) / 1024.0f;

            util::Vec3 fmin((INT)vmin.x / bound, (INT)vmin.y / bound, (INT)vmin.y / bound);
            fmin.Scale(bound);
            vmin = fmin;

            util::Vec3 fmax((INT)(vmax.x / bound), (INT)(vmax.y / bound), (INT)(vmax.y / bound));
            fmax.Scale(bound);
            vmax = fmax; */

            /*vmax = util::Mat4::Transform(lightCamera->GetIView(), vmax);
            vmin = util::Mat4::Transform(lightCamera->GetIView(), vmin); */
            /*util::Vec3 vDiagonal = vFrustumPoints[tbd::rightUpNear] - vFrustumPoints[tbd::leftDownFar];
            FLOAT l = vDiagonal.Length();
            vDiagonal.Set(l, l, l);
            // The offset calculated will pad the ortho projection so that it is always the same size 
            // and big enough to cover the entire cascade interval.
            util::Vec3 diff = vmax - vmin;
            util::Vec3 vBoarderOffset = vDiagonal - diff;
            vBoarderOffset.Scale(0.5f);

            vBoarderOffset.z = 0;
            // Add the offsets to the projection.
            vmax = vmax + vBoarderOffset;
            vmin = vmin - vBoarderOffset; */
            
            float n = min(-120, vmin.z); //todo
            float ff = vmax.z;

            distances[ci] = ff - n;

            XMMATRIX mat = XMMatrixOrthographicOffCenterLH(vmin.x, vmax.x, vmin.y, vmax.y, n, ff);

            XMStoreFloat4x4(&settings.m_projection.m_m, mat);

            /*
            util::StaticCamera staticCam(1,1,1,1);
#ifdef CSM_DEBUG
            staticCam.SetOrthographicProjectionOffCenter(vmin.x, vmax.x, vmin.y, vmax.y, n, ff);
            staticCam.SetView(lightCamera->GetView(), lightCamera->GetIView());
            ortographicFrusta = staticCam.GetFrustum();
#endif*/
            ortographicFrusta.CreateOrthographicOffCenter(vmin.x, vmax.x, vmin.y, vmax.y, n, ff);
            ortographicFrusta.Transform(lightCamera->GetIView());
   
            renderer->VPushProjectionTransform(settings.m_projection, ff - n);

            m_ppTargets[ci]->VClear();
            m_ppTargets[ci]->VBind();

            graph->VPushFrustum(&ortographicFrusta);
            m_pProgram->VBind();
            graph->VOnRender(CM_RENDERPATH_SHADOWMAP);

            m_pProgramInstanced->VBind();
            graph->VOnRender(CM_RENDERPATH_SHADOWMAP_INSTANCED);

            /*m_pProgramInstanced->VBind();
            graph->VOnRender(eRenderPath_DrawToShadowMapInstanced);*/
            graph->VPopFrustum();

            renderer->VPopProjectionTransform();

            m_ppBlurChain[ci]->VProcess();

         //   d3d::GetContext()->GenerateMips(m_ppBlurredTargets[ci]->GetShaderRessourceView());
        }

        renderer->VPopViewTransform();
        CmGetApp()->VGetHumanView()->VGetRenderer()->VGetCurrentRenderTarget()->VBind();

        IDeviceTexture* v[3];
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            v[i] = m_ppBlurredTargets[i]->VGetTexture();
        }
        renderer->VSetTextures(eEffect0, v, m_cascades);

        //ID3D11ShaderResourceView* debugView = m_ppBlurredTargets[0]->GetShaderRessourceView();
        //d3d::GetContext()->PSSetShaderResources(d3d::eEffect3, 1, &debugView); //debugging samplers
        
        util::Mat4 mats[3]; //TODO
        //FLOAT distances[3];
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            mats[i] = m_pCascadesSettings[i].m_projection;
        }

        IConstShaderBuffer* lb = renderer->VGetConstShaderBuffer(eEnvLightingBuffer);
        _LightingBuffer* _lb = (_LightingBuffer*)lb->VMap();
        _lb->m_view = lightCamera->GetView().m_m;
        _lb->m_iView = lightCamera->GetIView().m_m;
        _lb->m_projection[0] = mats[0].m_m;
        _lb->m_projection[1] = mats[1].m_m;
        _lb->m_projection[2] = mats[2].m_m;
        _lb->m_lightPos.x = lcc->GetCamera()->GetEyePos().x;
        _lb->m_lightPos.y = lcc->GetCamera()->GetEyePos().y;
        _lb->m_lightPos.z = lcc->GetCamera()->GetEyePos().z;
        _lb->m_intensity.x = m_intensity.x;
        _lb->m_intensity.y = m_intensity.y;
        _lb->m_intensity.z = m_intensity.z;
        _lb->m_ambient.x = m_ambient.x;
        _lb->m_ambient.y = m_ambient.y;
        _lb->m_ambient.z = m_ambient.z;
        _lb->m_distances.x = distances[0];
        _lb->m_distances.y = distances[1];
        _lb->m_distances.z = distances[2];
        lb->VUnmap();
       // renderer->SetCSMSettings(lightCamera->GetView(), lightCamera->GetIView(), mats, lcc->GetCamera()->GetEyePos(), distances);
    }

    void CascadedShadowMapper::SetSunPositionDelegate(chimera::IEventPtr data)
    {
        std::shared_ptr<chimera::SetSunPositionEvent> e = std::static_pointer_cast<chimera::SetSunPositionEvent>(data);
        CameraComponent* cmp;
        m_lightActorCamera->VQueryComponent(CM_CMP_CAMERA, (IActorComponent**)&cmp);
        util::Vec3 newPos = e->m_position;
        newPos.Normalize();
        newPos.Scale(1000.0f);
        cmp->GetCamera()->LookAt(newPos, util::Vec3(0,0,0));
//         util::Vec3 f(0,0,0);
//         util::Vec3 u(0,1,0);
//         util::Mat4 m;
//         XMStoreFloat4x4(&m.m_m, XMMatrixLookAtLH(XMLoadFloat3(&newPos.m_v), XMLoadFloat3(&(f.m_v)), XMLoadFloat3(&u.m_v)));
    }

    void CascadedShadowMapper::SetSunIntensityDelegate(chimera::IEventPtr data)
    {
        std::shared_ptr<SetSunIntensityEvent> event = std::static_pointer_cast<SetSunIntensityEvent>(data);
        m_intensity = event->m_intensity;
    }

    void CascadedShadowMapper::SetSunAmbientDelegate(chimera::IEventPtr data)
    {
        std::shared_ptr<SetSunAmbient> event = std::static_pointer_cast<SetSunAmbient>(data);
        m_ambient = event->m_ambient;
    }

    void CascadedShadowMapper::Destroy(void)
    {
        for(UCHAR i = 0; i < m_cascades; ++i)
        {
            SAFE_DELETE(m_ppTargets[i]);
            SAFE_DELETE(m_ppBlurredTargets[i]);
            SAFE_DELETE(m_ppBlurChain[i]);
        }

        SAFE_ARRAY_DELETE(m_ppTargets);

        SAFE_ARRAY_DELETE(m_ppBlurredTargets);

        SAFE_ARRAY_DELETE(m_ppBlurChain);

        SAFE_ARRAY_DELETE(m_pCascadesSettings);
    }

    CascadedShadowMapper::~CascadedShadowMapper(void)
    {
        REMOVE_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunPositionDelegate, CM_EVENT_SET_SUN_POSITION);
        REMOVE_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunIntensityDelegate, CM_EVENT_SET_SUN_INTENSITY);
        REMOVE_EVENT_LISTENER(this, &CascadedShadowMapper::SetSunAmbientDelegate, CM_EVENT_SET_SUN_AMBIENT);
        Destroy();
    }
};
