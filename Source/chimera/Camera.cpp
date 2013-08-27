#include "Camera.h"
#include "util.h"
#include <math.h>
#include "math.h"
#include "GameApp.h"
#include "Input.h"
namespace util 
{
    class IProjectionTypeHandler
    {
    public:
        virtual VOID VComputeProjection(Camera* camera) = 0;
    };

    class PerspectiveHandler : public IProjectionTypeHandler
    {
    public:
        VOID VComputeProjection(Camera* camera)
        {
            float invTan = 1.0f / tan(camera->m_FoV * 0.5f);

            camera->m_projection.m_m._11 = invTan;
            camera->m_projection.m_m._12 = 0;
            camera->m_projection.m_m._13 = 0;
            camera->m_projection.m_m._14 = 0;

            camera->m_projection.m_m._21 = 0;
            camera->m_projection.m_m._22 = invTan * camera->m_Aspect;
            camera->m_projection.m_m._23 = 0;
            camera->m_projection.m_m._24 = 0;

            camera->m_projection.m_m._31 = 0;
            camera->m_projection.m_m._32 = 0;
            camera->m_projection.m_m._33 = (camera->m_Far + camera->m_Near) / (camera->m_Far - camera->m_Near);
            camera->m_projection.m_m._34 = 1;

            camera->m_projection.m_m._41 = 0;
            camera->m_projection.m_m._42 = 0;
            camera->m_projection.m_m._43 = camera->m_Far  * (1 - camera->m_projection.m_m._33);
            camera->m_projection.m_m._44 = 0;

            camera->m_frustum.CreatePerspective(camera->m_Aspect, camera->m_FoV, camera->GetNear(), camera->GetFar());
        }
    };

    class OrthographicHandler : public IProjectionTypeHandler
    {
    public:
        VOID VComputeProjection(Camera* camera)
        {
            XMMATRIX mat = XMMatrixOrthographicLH(camera->m_width, camera->m_height, camera->GetNear(), camera->GetFar());
            XMFLOAT4X4 matf;
            XMStoreFloat4x4(&matf, mat);
            camera->m_projection.m_m = matf;
            camera->m_frustum.CreateOrthographic(camera->m_width, camera->m_height, camera->GetNear(), camera->GetFar());
        }
    };

    class OrthographicOffCenterHandler : public IProjectionTypeHandler
    {
    public:
        VOID VComputeProjection(Camera* camera)
        {
            XMMATRIX mat = XMMatrixOrthographicOffCenterLH(camera->m_left, camera->m_right, camera->m_down, camera->m_up, camera->GetNear(), camera->GetFar());
            XMFLOAT4X4 matf;
            XMStoreFloat4x4(&matf, mat);
            camera->m_projection.m_m = matf;
            camera->m_frustum.CreateOrthographicOffCenter(camera->m_left, camera->m_right, camera->m_down, camera->m_up, camera->GetNear(), camera->GetFar());
        }
    };

    Camera::Camera(UINT width, UINT height, FLOAT zNear, FLOAT zFar) 
    {
        m_height = (FLOAT)height;
        m_width = (FLOAT)width;
        this->m_Aspect = width / (FLOAT)height;
        this->m_FoV = XM_PIDIV2;
        this->m_Far = zFar;
        this->m_Near = zNear;
        this->m_Phi = 0;
        this->m_Theta = 0;
        this->m_eyePos = Vec3(0, 0, 0);
        this->m_viewDir = Vec3(0, 0, 1);
        this->m_upDir = Vec3(0, 1, 0);
        this->m_sideDir = Vec3(1, 0, 0);
        m_handler = NULL;
        SetProjectionType(ePerspective);
        this->ComputeProjection();
        this->ComputeView();
    }

    VOID Camera::Move(CONST Vec3& dt) {
         this->Move(dt.x, dt.y, dt.z);
    }

    VOID Camera::Move(FLOAT dx, FLOAT dy, FLOAT dz) {
         Vec3 deltaX(this->m_sideDir);
         deltaX.Scale(dx);

         Vec3 deltaZ(this->m_viewDir);
         deltaZ.Scale(dz);

         Vec3 deltaY(this->m_upDir);
         deltaY.Scale(dy);

         deltaX.Add(deltaY);
         deltaX.Add(deltaZ);

         this->m_eyePos.Add(deltaX); 

         this->ComputeView();
    }

    VOID Camera::Rotate(FLOAT dPhi, FLOAT dTheta) {
         this->m_Phi += dPhi;
         this->m_Theta += dTheta;

         FLOAT sinPhi = sin(this->m_Phi);
         FLOAT cosPhi = cos(this->m_Phi);
         FLOAT sinTheta = sin(this->m_Theta);
         FLOAT cosTheta = cos(this->m_Theta);
        
         this->m_sideDir.Set(cosPhi, 0, -sinPhi);
         this->m_upDir.Set(sinPhi*sinTheta, cosTheta, cosPhi*sinTheta);
         this->m_viewDir.Set(sinPhi*cosTheta, -sinTheta, cosPhi*cosTheta);

         this->ComputeView();
    }

    VOID Camera::MoveToPosition(CONST util::Vec3& pos) 
    {
        Vec3 delta = pos - this->m_lastSetPostition;
        Move(delta);
        this->m_lastSetPostition = pos;
    }

    VOID Camera::SetRotation(FLOAT phi, FLOAT theta) 
    {
        this->m_Phi = 0;
        this->m_Theta = 0;
        Rotate(phi, theta);
    }

    VOID Camera::FromViewUp(CONST util::Vec3& viewDir, CONST util::Vec3& viewUp)
    {
        m_upDir = viewUp;
        m_viewDir = viewDir;
        m_sideDir = util::Vec3::GetCross(m_upDir, m_upDir);
        ComputeView();
    }

    VOID Camera::LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at)
    {
        /*
        XMVECTOR v = {0,1,0};
        XMMATRIX m = XMMatrixLookAtLH(XMLoadFloat3(&eyePos.m_v), XMLoadFloat3(&at.m_v), v);
        
        m_eyePos = eyePos;
        util::Mat4 mm;
        XMStoreFloat4x4(&m_view.m_m, m);

        
        XMVECTOR d;
        XMMATRIX mi = XMMatrixInverse(&d, m);
        XMStoreFloat4x4(&m_iview.m_m, m); */
        
        util::Vec3 dir = at - eyePos;
        dir.Normalize();
        m_eyePos = eyePos;
        //FLOAT xz = sqrt(dir.x * dir.x + dir.z * dir.z);
        FLOAT phi = atan2(dir.x, dir.z); //-2.0f * XM_PI + (xz < 0.001f ? 0.0f : asin(dir.x / xz));
        FLOAT theta = -XM_PIDIV2 + acos(dir.y);

        SetRotation(phi, theta);
    }

    VOID Camera::SetEyePos(CONST util::Vec3& pos)
    {
        m_eyePos = pos;
        ComputeView();
    }

    VOID Camera::ComputeProjection(VOID) {
        m_handler->VComputeProjection(this);
    }

    VOID Camera::ComputeView(VOID) 
    {
         Vec3 zAxis = Vec3::GetNormalize(this->m_viewDir);
         Vec3 yAxis(this->m_upDir);
         Vec3 xAxis = Vec3::GetCross(yAxis, zAxis);
         xAxis.Normalize();
         yAxis = Vec3::GetCross(zAxis, xAxis);
         //util::Vec3 focusPoint = zAxis + m_eyePos;
        //m_view.m_m = Mat4::GetFromMat4x4(XMMatrixLookAtLH(Vec3::GetFromFloat3(m_eyePos.m_v), Vec3::GetFromFloat3(focusPoint.m_v), Vec3::GetFromFloat3(yAxis.m_v)));
    
         this->m_view.m_m._11 = xAxis.x;
         this->m_view.m_m._21 = xAxis.y;
         this->m_view.m_m._31 = xAxis.z;
         this->m_view.m_m._41 = -(this->m_eyePos.x * xAxis.x + this->m_eyePos.y * xAxis.y + this->m_eyePos.z * xAxis.z);
     
         this->m_view.m_m._12 = yAxis.x;
         this->m_view.m_m._22 = yAxis.y;
         this->m_view.m_m._32 = yAxis.z;
         this->m_view.m_m._42 = -(this->m_eyePos.x * yAxis.x + this->m_eyePos.y * yAxis.y + this->m_eyePos.z * yAxis.z);

         this->m_view.m_m._13 = zAxis.x;
         this->m_view.m_m._23 = zAxis.y;
         this->m_view.m_m._33 = zAxis.z;
         this->m_view.m_m._43 = -(this->m_eyePos.x * zAxis.x + this->m_eyePos.y * zAxis.y + this->m_eyePos.z * zAxis.z);

         this->m_view.m_m._14 = 0;
         this->m_view.m_m._24 = 0;
         this->m_view.m_m._34 = 0;
         this->m_view.m_m._44 = 1; 

         this->m_iview.m_m._11 = xAxis.x;
         this->m_iview.m_m._12 = xAxis.y;
         this->m_iview.m_m._13 = xAxis.z;
         this->m_iview.m_m._41 = this->m_eyePos.x;

         this->m_iview.m_m._21 = yAxis.x;
         this->m_iview.m_m._22 = yAxis.y;
         this->m_iview.m_m._23 = yAxis.z;
         this->m_iview.m_m._42 = this->m_eyePos.y;

         this->m_iview.m_m._31 = zAxis.x;
         this->m_iview.m_m._32 = zAxis.y;
         this->m_iview.m_m._33 = zAxis.z;
         this->m_iview.m_m._43 = this->m_eyePos.z;

         this->m_iview.m_m._14 = 0;
         this->m_iview.m_m._24 = 0;
         this->m_iview.m_m._34 = 0;
         this->m_iview.m_m._44 = 1;

         m_frustum.Transform(m_iview);

        //util::Mat4::Mul(m_iview, m_view).Print();
         this->m_viewProjection = util::Mat4::Mul(m_projection, m_view);
    }

    VOID Camera::SetProjectionType(ProjectionType type)
    {
        SAFE_DELETE(m_handler);
        switch(type)
        {
        case eOrthographic:
            {
                m_handler = new OrthographicHandler;
            } break;
        case ePerspective: 
            {
                m_handler = new PerspectiveHandler;
            } break;

        case OrthographicOffCenter:
            {
                m_handler = new OrthographicOffCenterHandler;
            } break;
        default : 
            {
                LOG_CRITICAL_ERROR("unknown type");
            } break;
        }
    }

    tbd::Frustum& Camera::GetFrustum(VOID)
    {
        return m_frustum;
    }

    VOID Camera::SetPerspectiveProjection(FLOAT aspect, FLOAT fov, FLOAT fnear, FLOAT ffar)
    {
        m_FoV = fov;
        m_Aspect = aspect;
        m_Near = fnear;
        m_Far = ffar;
        SetProjectionType(ePerspective);
        ComputeProjection();
    }

    VOID Camera::SetOrthographicProjection(FLOAT width, FLOAT height, FLOAT fnear, FLOAT ffar)
    {
        m_width = width;
        m_height = height;
        m_Far = ffar;
        m_Near = fnear;
        SetProjectionType(eOrthographic);
        ComputeProjection();
    }

    VOID Camera::SetOrthographicProjectionOffCenter(FLOAT left, FLOAT right, FLOAT down, FLOAT up, FLOAT fNear, FLOAT fFar)
    {
        m_left = left;
        m_right = right;
        m_down = down;
        m_up = up;
        m_Far = fFar;
        m_Near = fNear;
        SetProjectionType(OrthographicOffCenter);
        ComputeProjection();
    }

    Camera::~Camera(VOID) 
    {
        SAFE_DELETE(m_handler);
    }

    FPSCamera::FPSCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar) :  Camera(width, height, zNear, zFar)
    {

    }

    VOID FPSCamera::Rotate(FLOAT dPhi, FLOAT dTheta)
    {
       FLOAT newPhi = m_Phi + dPhi;
       FLOAT newTheta = m_Theta + dTheta;
       newTheta = CLAMP(newTheta, -XM_PIDIV2, XM_PIDIV2);
       m_Phi = 0;
       m_Theta = 0;
       Camera::Rotate(newPhi, newTheta);
    }

    FPSCamera::~FPSCamera(VOID)
    {

    }

    CharacterCamera::CharacterCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar) : FPSCamera(width, height, zNear, zFar), m_yOffset(0)
    {

    }

    VOID CharacterCamera::SetEyePos(CONST util::Vec3& pos)
    {
        FPSCamera::SetEyePos(pos + util::Vec3(0, m_yOffset, 0));
    }

    VOID CharacterCamera::LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at)
    {
        FPSCamera::LookAt(eyePos + util::Vec3(0, m_yOffset, 0), at);
    }

    VOID CharacterCamera::MoveToPosition(CONST util::Vec3& pos)
    {
        this->m_eyePos.Set(pos.x, pos.y + m_yOffset, pos.z);
        ComputeView();
    }

    CharacterHeadShake::CharacterHeadShake(UINT width, UINT height, FLOAT zNear, FLOAT zFar) : CharacterCamera(width, height, zNear, zFar)
    {

    }

    VOID CharacterHeadShake::MoveToPosition(CONST util::Vec3& pos)
    {

        INT x = tbd::math::sign(abs(pos.x - m_lastPos.x));
        INT y = tbd::math::sign(abs(pos.z - m_lastPos.z));

        if(x != 0 || y != 0)
        {
            FLOAT speed = app::g_pApp->GetInputHandler()->IsKeyDown(KEY_LSHIFT) ? 1.75f : 1.0f;
            util::Vec3 headShake;
            headShake.x = (FLOAT)cos(speed * m_timer.GetTime() * 1e-2); 
            headShake.y = 1 - headShake.x * headShake.x;
            m_headShake = headShake * 1.5e-1f;
            m_lastPos = pos;
            m_timer.Tick();
        }

        CharacterCamera::MoveToPosition(pos);
        FPSCamera::Move(m_headShake);
    }

    VOID CharacterHeadShake::Rotate(FLOAT dPhi, FLOAT dTheta) 
    {
        CharacterCamera::Rotate(dPhi, dTheta);

        FLOAT sinPhi = sin(this->m_Phi);
        FLOAT cosPhi = cos(this->m_Phi);

        m_modViewDir.Set(sinPhi, 0, cosPhi);
    }

    CONST util::Vec3& CharacterHeadShake::GetViewDirXZ(VOID)
    {
        return m_modViewDir;
    }

    StaticCamera::StaticCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar) : FPSCamera(width, height, zNear, zFar)
    {

    }

    VOID StaticCamera::Move(FLOAT dx, FLOAT dy, FLOAT dz)
    {

    }

    VOID StaticCamera::Move(CONST Vec3& dt)
    {

    }

    VOID StaticCamera::Rotate(FLOAT dPhi, FLOAT dTheta)
    {

    }

    VOID StaticCamera::SetView(CONST util::Mat4& view, CONST util::Mat4& iview)
    {
        m_view = view;
        m_iview = iview;
        m_frustum.Transform(iview);
    }
}
