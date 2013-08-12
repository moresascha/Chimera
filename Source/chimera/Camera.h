#pragma once
#include "stdafx.h"
#include "Mat4.h"
#include "Vec3.h"
#include "Frustum.h"
#include "Timer.h"

namespace util 
{
    enum ProjectionType
    {
        ePerspective,
        eOrthographic,
        OrthographicOffCenter
    };

    class ICamera 
    {
    public:
        virtual CONST Mat4& GetView(VOID) = 0;

        virtual CONST Mat4& GetIView(VOID) = 0;

        virtual CONST Mat4& GetProjection(VOID) = 0;

        virtual CONST Mat4& GetViewProjection(VOID) = 0;

        virtual CONST Vec3& GetEyePos(VOID) = 0;

        virtual CONST Vec3& GetViewDir(VOID) = 0;

        virtual CONST Vec3& GetViewDirXZ(VOID) = 0;

        virtual CONST Vec3& GetSideDir(VOID) = 0;

        virtual CONST Vec3& GetUpDir(VOID) = 0;

        virtual FLOAT GetPhi(VOID) CONST = 0;

        virtual FLOAT GetTheta(VOID) CONST = 0;

        virtual VOID SetProjectionType(ProjectionType type) = 0;

        virtual VOID LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at) = 0;

        virtual VOID SetEyePos(CONST util::Vec3& pos) = 0;

        virtual VOID Move(FLOAT dx, FLOAT dy, FLOAT dz) {}

        virtual VOID Move(CONST Vec3& dt) {}

        virtual VOID Rotate(FLOAT dPhi, FLOAT dTheta) {}

        virtual VOID SetAspect(UINT width, UINT height) {}

        virtual FLOAT GetAspect(VOID) { return 0; }

        virtual VOID SetFoV(FLOAT foV) {}

        virtual FLOAT GetFoV(VOID) { return 0; }

        virtual VOID SetFar(FLOAT f) {}

        virtual FLOAT GetNear(VOID) { return 0; }

        virtual VOID SetNear(FLOAT n){}

        virtual FLOAT GetFar(VOID) { return 0; }

        virtual VOID MoveToPosition(CONST util::Vec3& pos) {}

        virtual VOID SetRotation(FLOAT phi, FLOAT theta) {}

        virtual VOID VSetYOffset(FLOAT offset) { }

        virtual VOID SetPerspectiveProjection(FLOAT aspect, FLOAT fov, FLOAT fnear, FLOAT ffar) {}

        virtual VOID SetOrthographicProjection(FLOAT width, FLOAT height, FLOAT fnear, FLOAT ffar) {}

        virtual VOID SetPrthographicProjectionOffCenter(FLOAT left, FLOAT right, FLOAT up, FLOAT down, FLOAT fNear, FLOAT fFar) {}

        virtual tbd::Frustum& GetFrustum(VOID) = 0;

        virtual ~ICamera(VOID) {}
    };

    class IProjectionTypeHandler;
    class PerspectiveHandler;
    class OrhtographicHandler;

    class DLL_EXPORT Camera : public ICamera
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
        tbd::Frustum m_frustum;

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

        VOID SetPerspectiveProjection(FLOAT aspect, FLOAT fov, FLOAT fnear, FLOAT ffar);

        VOID SetOrthographicProjection(FLOAT width, FLOAT height, FLOAT fnear, FLOAT ffar);

        VOID SetOrthographicProjectionOffCenter(FLOAT left, FLOAT right, FLOAT down, FLOAT up, FLOAT fNear, FLOAT fFar);

        VOID SetAspect(UINT width, UINT height) 
        {
            this->m_Aspect = width / (FLOAT) height;
            m_height = (FLOAT)height;
            m_width = (FLOAT)width;
            this->ComputeProjection();
        }

        FLOAT GetAspect(VOID)
        {
            return m_Aspect;
        }

        VOID SetFoV(FLOAT foV) {
            this->m_FoV = foV;
            this->ComputeProjection();
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

        tbd::Frustum& GetFrustum(VOID);

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

