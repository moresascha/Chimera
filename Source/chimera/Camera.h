#pragma once
#include "stdafx.h"
#include "Mat4.h"
#include "Vec3.h"
#include "Frustum.h"
#include "Timer.h"

namespace chimera
{
    namespace util 
    {
        class IProjectionTypeHandler;
        class PerspectiveHandler;
        class OrhtographicHandler;

        class Camera : public ICamera
        {
            friend class PerspectiveHandler;
            friend class OrthographicHandler;
            friend class OrthographicOffCenterHandler;
        protected:
            FLOAT m_Aspect;
            FLOAT m_FoV;
            FLOAT m_Near;
            FLOAT m_Far;
            FLOAT m_Phi;
            FLOAT m_Theta;
            FLOAT m_width;
            FLOAT m_height;
            FLOAT m_left;
            FLOAT m_right;
            FLOAT m_up;
            FLOAT m_down;
            Mat4 m_view;
            Mat4 m_iview;
            Mat4 m_projection;
            Mat4 m_viewProjection;
            Vec3 m_eyePos;
            Vec3 m_sideDir;
            Vec3 m_upDir;
            Vec3 m_viewDir;
            Vec3 m_lastSetPostition;
            VOID ComputeProjection(VOID);
            VOID ComputeView(VOID);
            IProjectionTypeHandler* m_handler;
            chimera::Frustum m_frustum;

        public:
            Camera(UINT width, UINT height, FLOAT zNear, FLOAT zFar);

            CONST Mat4& GetView(VOID) 
            {
                return this->m_view;
            }

            CONST Mat4& GetIView(VOID)
            {
                return m_iview;
            }

            CONST Mat4& GetProjection(VOID) 
            {
                return this->m_projection;
            }

            CONST Mat4& GetViewProjection(VOID)
            {
                return this->m_viewProjection;
            }

            CONST Vec3& GetEyePos(VOID) 
            {
                return this->m_eyePos;
            }

            CONST Vec3& GetViewDir(VOID) 
            {
                return this->m_viewDir;
            }

            virtual CONST Vec3& GetViewDirXZ(VOID) 
            {
                return this->m_viewDir;
            }

            CONST Vec3& GetSideDir(VOID)
            {
                return m_sideDir;
            }

            CONST Vec3& GetUpDir(VOID)
            {
                return m_upDir;
            }

            FLOAT GetPhi(VOID) CONST
            {
                return m_Phi;
            }

            FLOAT GetTheta(VOID) CONST
            {
                return m_Theta;
            }

            virtual VOID Move(FLOAT dx, FLOAT dy, FLOAT dz);

            virtual VOID Move(CONST Vec3& dt);

            virtual VOID Rotate(FLOAT dPhi, FLOAT dTheta);

            virtual VOID LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at);

            VOID FromViewUp(CONST util::Vec3& viewDir, CONST util::Vec3& viewUp);

            VOID SetPerspectiveProjection(FLOAT aspect, FLOAT fov, FLOAT fnear, FLOAT ffar);

            VOID SetOrthographicProjection(FLOAT width, FLOAT height, FLOAT fnear, FLOAT ffar);

            VOID SetOrthographicProjectionOffCenter(FLOAT left, FLOAT right, FLOAT down, FLOAT up, FLOAT fNear, FLOAT fFar);

            VOID SetAspect(UINT width, UINT height) 
            {
                m_Aspect = width / (FLOAT) height;
                m_height = (FLOAT)height;
                m_width = (FLOAT)width;
                ComputeProjection();
            }

            FLOAT GetAspect(VOID)
            {
                return m_Aspect;
            }

            VOID SetFoV(FLOAT foV) {
                m_FoV = foV;
                ComputeProjection();
            }

            FLOAT GetFoV(VOID)
            {
                return m_FoV;
            }

            VOID SetFar(FLOAT f) 
            {
                this->m_Far = f;
                this->ComputeProjection();
            }

            FLOAT GetFar(VOID)
            {
                return m_Far;
            }

            VOID SetNear(FLOAT n) 
            {
                this->m_Near = n;
                this->ComputeProjection();
            }

            FLOAT GetNear(VOID)
            {
                return m_Near;
            }

            virtual VOID MoveToPosition(CONST util::Vec3& pos);

            virtual VOID SetEyePos(CONST util::Vec3& pos);

            VOID SetRotation(FLOAT phi, FLOAT theta);

            VOID SetProjectionType(ProjectionType type);

            chimera::Frustum& GetFrustum(VOID);

            virtual ~Camera(VOID);
        };

        class FPSCamera : public Camera
        {
        public:
            FPSCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar);
            virtual VOID Rotate(FLOAT dPhi, FLOAT dTheta);
            virtual ~FPSCamera();
        };

        class CharacterCamera : public FPSCamera
        {
        private:
            FLOAT m_yOffset;
        public:
            CharacterCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar);
            VOID VSetYOffset(FLOAT offset) { m_yOffset = offset; }
            VOID SetEyePos(CONST util::Vec3& pos);
            VOID LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at);
            virtual VOID MoveToPosition(CONST util::Vec3& pos);
        };

        class CharacterHeadShake : public CharacterCamera
        {
        private:
            util::Timer m_timer;
            util::Vec3 m_headShake;
            util::Vec3 m_lastPos;
            util::Vec3 m_modViewDir;
        public:
            CharacterHeadShake(UINT width, UINT height, FLOAT zNear, FLOAT zFar);
            VOID MoveToPosition(CONST util::Vec3& pos);
            CONST Vec3& GetViewDirXZ(VOID);
            VOID Rotate(FLOAT dPhi, FLOAT dTheta);
        };

        class StaticCamera : public FPSCamera
        {

        public:

            StaticCamera(UINT width, UINT height, FLOAT zNear, FLOAT zFar);

            VOID Move(FLOAT dx, FLOAT dy, FLOAT dz);

            VOID Move(CONST Vec3& dt);

            VOID Rotate(FLOAT dPhi, FLOAT dTheta);

            VOID SetView(CONST util::Mat4& view, CONST util::Mat4& iview);
        };
    }
}

