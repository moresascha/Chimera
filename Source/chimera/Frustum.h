#pragma once
#include "stdafx.h"
#include "Plane.h"
#include "Vec3.h"
#include "AxisAlignedBB.h"

namespace tbd
{
    enum Planes {left, right, up, down, front, back};
    enum Points {rightUpNear, rightDownNear, leftDownNear, leftUpNear, rightUpFar, rightDownFar, leftDownFar, leftUpFar};
    class Frustum
    {
    protected:
        util::Plane m_planes[6];
        util::Vec3 m_points[8];
        util::Vec3 m_Initpoints[8];

        VOID Create(FLOAT l, FLOAT r, FLOAT u, FLOAT d, FLOAT fNear, FLOAT fFar);
        VOID CreatePlanes(VOID);
    public:
        Frustum(VOID) {}
        virtual VOID CreatePerspective(FLOAT aspect, FLOAT fov, FLOAT fNear, FLOAT fFar);
        virtual VOID CreateOrthographic(FLOAT width, FLOAT height, FLOAT fNear, FLOAT fFar);
        virtual VOID CreateOrthographicOffCenter(FLOAT left, FLOAT right, FLOAT bottom, FLOAT top, FLOAT fNear, FLOAT fFar);

        virtual BOOL IsInside(CONST util::Vec3& point) CONST;
        virtual BOOL IsInside(CONST util::Vec3& point, FLOAT radius) CONST;
        virtual BOOL IsInside(CONST util::AxisAlignedBB& aabb) CONST;
        virtual VOID Transform(CONST util::Mat4& mat);
        CONST util::Vec3* GetPoints(VOID) CONST; 
        virtual ~Frustum(VOID) {}
    };

    class PointLightFrustum : public Frustum
    {
        Frustum m_frustums[6];
        util::Mat4 m_mats[6];
        FLOAT m_radius;
        util::Vec3 m_position;
    public:
        PointLightFrustum(VOID) {};
        VOID SetMatrices(util::Mat4 mats[6]);
        VOID CreatePerspective(FLOAT aspect, FLOAT fov, FLOAT fNear, FLOAT fFar);
        //virtual BOOL IsInside(CONST util::Vec3& point) CONST;
        VOID SetParams(FLOAT radius, CONST util::Vec3& pos);
        BOOL IsInside(CONST util::Vec3& point, FLOAT radius) CONST;
        BOOL IsInside(CONST util::AxisAlignedBB& aabb) CONST;
        VOID Transform(CONST util::Mat4& mat);
    };
}
