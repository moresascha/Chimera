#include "Camera.h"
#include "util.h"
#include <math.h>
#include "cm_math.h"
namespace chimera
{
    namespace util 
    {
        class IProjectionTypeHandler
        {
        public:
            virtual void VComputeProjection(Camera* camera) = 0;
        };

        class PerspectiveHandler : public IProjectionTypeHandler
        {
        public:
            void VComputeProjection(Camera* camera)
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
            void VComputeProjection(Camera* camera)
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
            void VComputeProjection(Camera* camera)
            {
                XMMATRIX mat = XMMatrixOrthographicOffCenterLH(camera->m_left, camera->m_right, camera->m_down, camera->m_up, camera->GetNear(), camera->GetFar());
                XMFLOAT4X4 matf;
                XMStoreFloat4x4(&matf, mat);
                camera->m_projection.m_m = matf;
                camera->m_frustum.CreateOrthographicOffCenter(camera->m_left, camera->m_right, camera->m_down, camera->m_up, camera->GetNear(), camera->GetFar());
            }
        };

        Camera::Camera(uint width, uint height, float zNear, float zFar) 
        {
            m_height = (float)height;
            m_width = (float)width;
            this->m_Aspect = width / (float)height;
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
            SetProjectionType(eProjectionType_Perspective);
            this->ComputeProjection();
            this->ComputeView();
        }

        void Camera::Move(const Vec3& dt) {
             this->Move(dt.x, dt.y, dt.z);
        }

        void Camera::Move(float dx, float dy, float dz) {
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

        void Camera::Rotate(float dPhi, float dTheta) {
             this->m_Phi += dPhi;
             this->m_Theta += dTheta;

             float sinPhi = sin(this->m_Phi);
             float cosPhi = cos(this->m_Phi);
             float sinTheta = sin(this->m_Theta);
             float cosTheta = cos(this->m_Theta);
        
             this->m_sideDir.Set(cosPhi, 0, -sinPhi);
             this->m_upDir.Set(sinPhi*sinTheta, cosTheta, cosPhi*sinTheta);
             this->m_viewDir.Set(sinPhi*cosTheta, -sinTheta, cosPhi*cosTheta);

             this->ComputeView();
        }

        void Camera::MoveToPosition(const util::Vec3& pos) 
        {
            Vec3 delta = pos - this->m_lastSetPostition;
            Move(delta);
            this->m_lastSetPostition = pos;
        }

        void Camera::SetRotation(float phi, float theta) 
        {
            this->m_Phi = 0;
            this->m_Theta = 0;
            Rotate(phi, theta);
        }

        void Camera::FromViewUp(const util::Vec3& viewDir, const util::Vec3& viewUp)
        {
            m_upDir = viewUp;
            m_viewDir = viewDir;
            m_sideDir = util::Vec3::GetCross(m_upDir, m_upDir);
            ComputeView();
        }

        void Camera::LookAt(const util::Vec3& eyePos, const util::Vec3& at)
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
            float phi = atan2(dir.x, dir.z); //-2.0f * XM_PI + (xz < 0.001f ? 0.0f : asin(dir.x / xz));
            float theta = -XM_PIDIV2 + acos(dir.y);

            SetRotation(phi, theta);
        }

        void Camera::SetEyePos(const util::Vec3& pos)
        {
            m_eyePos = pos;
            ComputeView();
        }

        void Camera::ComputeProjection(void) {
            m_handler->VComputeProjection(this);
        }

        void Camera::ComputeView(void) 
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

        void Camera::SetProjectionType(ProjectionType type)
        {
            SAFE_DELETE(m_handler);
            switch(type)
            {
            case eProjectionType_Orthographic:
                {
                    m_handler = new OrthographicHandler;
                } break;
            case eProjectionType_Perspective: 
                {
                    m_handler = new PerspectiveHandler;
                } break;

            case eProjectionType_OrthographicOffCenter:
                {
                    m_handler = new OrthographicOffCenterHandler;
                } break;
            default : 
                {
                    LOG_CRITICAL_ERROR("unknown type");
                } break;
            }
        }

        chimera::Frustum& Camera::GetFrustum(void)
        {
            return m_frustum;
        }

        void Camera::SetPerspectiveProjection(float aspect, float fov, float fnear, float ffar)
        {
            m_FoV = fov;
            m_Aspect = aspect;
            m_Near = fnear;
            m_Far = ffar;
            SetProjectionType(eProjectionType_Perspective);
            ComputeProjection();
        }

        void Camera::SetOrthographicProjection(float width, float height, float fnear, float ffar)
        {
            m_width = width;
            m_height = height;
            m_Far = ffar;
            m_Near = fnear;
            SetProjectionType(eProjectionType_Orthographic);
            ComputeProjection();
        }

        void Camera::SetOrthographicProjectionOffCenter(float left, float right, float down, float up, float fNear, float fFar)
        {
            m_left = left;
            m_right = right;
            m_down = down;
            m_up = up;
            m_Far = fFar;
            m_Near = fNear;
            SetProjectionType(eProjectionType_OrthographicOffCenter);
            ComputeProjection();
        }

        Camera::~Camera(void) 
        {
            SAFE_DELETE(m_handler);
        }

        FPSCamera::FPSCamera(uint width, uint height, float zNear, float zFar) :  Camera(width, height, zNear, zFar)
        {

        }

        void FPSCamera::Rotate(float dPhi, float dTheta)
        {
           float newPhi = m_Phi + dPhi;
           float newTheta = m_Theta + dTheta;
           newTheta = CLAMP(newTheta, -XM_PIDIV2, XM_PIDIV2);
           m_Phi = 0;
           m_Theta = 0;
           Camera::Rotate(newPhi, newTheta);
        }

        FPSCamera::~FPSCamera(void)
        {

        }

        CharacterCamera::CharacterCamera(uint width, uint height, float zNear, float zFar) : FPSCamera(width, height, zNear, zFar), m_yOffset(1.75f)
        {

        }

        void CharacterCamera::SetEyePos(const util::Vec3& pos)
        {
            FPSCamera::SetEyePos(pos + util::Vec3(0, m_yOffset, 0));
        }

        void CharacterCamera::LookAt(const util::Vec3& eyePos, const util::Vec3& at)
        {
            FPSCamera::LookAt(eyePos + util::Vec3(0, m_yOffset, 0), at);
        }

        void CharacterCamera::MoveToPosition(const util::Vec3& pos)
        {
            this->m_eyePos.Set(pos.x, pos.y + m_yOffset, pos.z);
            ComputeView();
        }

        CharacterHeadShake::CharacterHeadShake(uint width, uint height, float zNear, float zFar) : CharacterCamera(width, height, zNear, zFar)
        {

        }

        void CharacterHeadShake::MoveToPosition(const util::Vec3& pos)
        {

            int x = chimera::math::sign(abs(pos.x - m_lastPos.x));
            int y = chimera::math::sign(abs(pos.z - m_lastPos.z));

            if(x != 0 || y != 0)
            {
                float speed = CmGetApp()->VGetInputHandler()->VIsKeyDown(KEY_LSHIFT) ? 1.75f : 1.0f;
                util::Vec3 headShake;
                headShake.x = (float)cos(speed * m_timer.VGetTime() * 1e-2); 
                headShake.y = 1 - headShake.x * headShake.x;
                m_headShake = headShake * 1.0e-1f;
                m_lastPos = pos;
                m_timer.VTick();
            }

            CharacterCamera::MoveToPosition(pos);
            FPSCamera::Move(m_headShake);
        }

        void CharacterHeadShake::Rotate(float dPhi, float dTheta) 
        {
            CharacterCamera::Rotate(dPhi, dTheta);

            float sinPhi = sin(m_Phi);
            float cosPhi = cos(m_Phi);

            m_modViewDir.Set(sinPhi, 0, cosPhi);
        }

        const util::Vec3& CharacterHeadShake::GetViewDirXZ(void)
        {
            return m_modViewDir;
        }

        StaticCamera::StaticCamera(uint width, uint height, float zNear, float zFar) : FPSCamera(width, height, zNear, zFar)
        {

        }

        void StaticCamera::Move(float dx, float dy, float dz)
        {

        }

        void StaticCamera::Move(const Vec3& dt)
        {

        }

        void StaticCamera::Rotate(float dPhi, float dTheta)
        {

        }

        void StaticCamera::SetView(const util::Mat4& view, const util::Mat4& iview)
        {
            m_view = view;
            m_iview = iview;
            m_frustum.Transform(iview);
        }
    }
}