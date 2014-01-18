#pragma once
#include "stdafx.h"
#include "Plane.h"
#include "Vec3.h"
#include "AxisAlignedBB.h"

namespace chimera
{
    enum Planes {left, right, up, down, front, back};
    enum Points {rightUpNear, rightDownNear, leftDownNear, leftUpNear, rightUpFar, rightDownFar, leftDownFar, leftUpFar};
    class Frustum
    {
    protected:
        util::Plane m_planes[6];
        util::Vec3 m_points[8];
        util::Vec3 m_Initpoints[8];

        void Create(float l, float r, float u, float d, float fNear, float fFar);
        void CreatePlanes(void);
    public:
        Frustum(void) {}
        virtual void CreatePerspective(float aspect, float fov, float fNear, float fFar);
        virtual void CreateOrthographic(float width, float height, float fNear, float fFar);
        virtual void CreateOrthographicOffCenter(float left, float right, float bottom, float top, float fNear, float fFar);

        virtual bool IsInside(const util::Vec3& point) const;
        virtual bool IsInside(const util::Vec3& point, float radius) const;
        virtual bool IsInside(const util::AxisAlignedBB& aabb) const;
        virtual void Transform(const util::Mat4& mat);
        const util::Vec3* GetPoints(void) const; 
        virtual ~Frustum(void) {}
    };

    class PointLightFrustum : public Frustum
    {
        Frustum m_frustums[6];
        util::Mat4 m_mats[6];
        float m_radius;
        util::Vec3 m_position;
    public:
        PointLightFrustum(void) {};
        void SetMatrices(util::Mat4 mats[6]);
        void CreatePerspective(float aspect, float fov, float fNear, float fFar);
        //virtual BOOL IsInside(CONST util::Vec3& point) CONST;
        void SetParams(float radius, const util::Vec3& pos);
        bool IsInside(const util::Vec3& point, float radius) const;
        bool IsInside(const util::AxisAlignedBB& aabb) const;
        void Transform(const util::Mat4& mat);
    };
}
