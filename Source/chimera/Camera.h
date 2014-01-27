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
            float m_Aspect;
            float m_FoV;
            float m_Near;
            float m_Far;
            float m_Phi;
            float m_Theta;
            float m_width;
            float m_height;
            float m_left;
            float m_right;
            float m_up;
            float m_down;
            Mat4 m_view;
            Mat4 m_iview;
            Mat4 m_projection;
            Mat4 m_viewProjection;
            Vec3 m_eyePos;
            Vec3 m_sideDir;
            Vec3 m_upDir;
            Vec3 m_viewDir;
            Vec3 m_lastSetPostition;
            void ComputeProjection(void);
            void ComputeView(void);
            IProjectionTypeHandler* m_handler;
            chimera::Frustum m_frustum;

        public:
            Camera(uint width, uint height, float zNear, float zFar);

            const Mat4& GetView(void) 
            {
                return this->m_view;
            }

            const Mat4& GetIView(void)
            {
                return m_iview;
            }

            const Mat4& GetProjection(void) 
            {
                return this->m_projection;
            }

            const Mat4& GetViewProjection(void)
            {
                return this->m_viewProjection;
            }

            const Vec3& GetEyePos(void) 
            {
                return this->m_eyePos;
            }

            const Vec3& GetViewDir(void) 
            {
                return this->m_viewDir;
            }

            virtual const Vec3& GetViewDirXZ(void) 
            {
                return this->m_viewDir;
            }

            const Vec3& GetSideDir(void)
            {
                return m_sideDir;
            }

            const Vec3& GetUpDir(void)
            {
                return m_upDir;
            }

            float GetPhi(void) const
            {
                return m_Phi;
            }

            float GetTheta(void) const
            {
                return m_Theta;
            }

            virtual void Move(float dx, float dy, float dz);

            virtual void Move(const Vec3& dt);

            virtual void Rotate(float dPhi, float dTheta);

            virtual void LookAt(const util::Vec3& eyePos, const util::Vec3& at);

            void FromViewUp(const util::Vec3& viewDir, const util::Vec3& viewUp);

            void SetPerspectiveProjection(float aspect, float fov, float fnear, float ffar);

            void SetOrthographicProjection(float width, float height, float fnear, float ffar);

            void SetOrthographicProjectionOffCenter(float left, float right, float down, float up, float fNear, float fFar);

            void SetAspect(uint width, uint height) 
            {
                m_Aspect = width / (float) height;
                m_height = (float)height;
                m_width = (float)width;
                ComputeProjection();
            }

            float GetAspect(void)
            {
                return m_Aspect;
            }

            void SetFoV(float foV) {
                m_FoV = foV;
                ComputeProjection();
            }

            float GetFoV(void)
            {
                return m_FoV;
            }

            void SetFar(float f) 
            {
                this->m_Far = f;
                this->ComputeProjection();
            }

            float GetFar(void)
            {
                return m_Far;
            }

            void SetNear(float n) 
            {
                this->m_Near = n;
                this->ComputeProjection();
            }

            float GetNear(void)
            {
                return m_Near;
            }

            virtual void MoveToPosition(const util::Vec3& pos);

            virtual void SetEyePos(const util::Vec3& pos);

            void SetRotation(float phi, float theta);

            void SetProjectionType(ProjectionType type);

            chimera::Frustum& GetFrustum(void);

            virtual ~Camera(void);
        };

        class FPSCamera : public Camera
        {
        public:
            FPSCamera(uint width, uint height, float zNear, float zFar);
            virtual void Rotate(float dPhi, float dTheta);
            virtual ~FPSCamera();
        };

        class CharacterCamera : public FPSCamera
        {
        private:
            float m_yOffset;
        public:
            CharacterCamera(uint width, uint height, float zNear, float zFar);
            void SetYOffset(float offset) { m_yOffset = offset; }
            float GetYOffset(void) const { return m_yOffset; }
            void SetEyePos(const util::Vec3& pos);
            void LookAt(const util::Vec3& eyePos, const util::Vec3& at);
            virtual void MoveToPosition(const util::Vec3& pos);
        };

        class CharacterHeadShake : public CharacterCamera
        {
        private:
            util::Timer m_timer;
            util::Vec3 m_headShake;
            util::Vec3 m_lastPos;
            util::Vec3 m_modViewDir;
        public:
            CharacterHeadShake(uint width, uint height, float zNear, float zFar);
            void MoveToPosition(const util::Vec3& pos);
            const Vec3& GetViewDirXZ(void);
            void Rotate(float dPhi, float dTheta);
        };

        class StaticCamera : public FPSCamera
        {

        public:

            StaticCamera(uint width, uint height, float zNear, float zFar);

            void Move(float dx, float dy, float dz);

            void Move(const Vec3& dt);

            void Rotate(float dPhi, float dTheta);

            void SetView(const util::Mat4& view, const util::Mat4& iview);
        };
    }
}

