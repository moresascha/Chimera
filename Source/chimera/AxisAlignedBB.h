#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Mat4.h"
#include "Line.h"

namespace chimera
{
    namespace util 
    {
        class AxisAlignedBB
        {
        private:
            util::Vec3 m_min;
            util::Vec3 m_max;
            util::Vec3 m_mid;
            util::Vec3 m_Tpoints[8];
            util::Line m_lines[12];
            FLOAT m_radius;
            util::Vec3 m_points[8];

        public:
            CM_DLL_API AxisAlignedBB(VOID);

            CM_DLL_API AxisAlignedBB(CONST AxisAlignedBB& aabb);

            CM_DLL_API VOID Clear(VOID);

            CM_DLL_API VOID AddPoint(CONST util::Vec3& point);

            CM_DLL_API VOID Construct(VOID);

            CM_DLL_API CONST util::Vec3& GetMin(VOID) CONST;

            CM_DLL_API CONST util::Vec3& GetMax(VOID) CONST;

            CM_DLL_API FLOAT GetMinAxis(UCHAR axis) CONST;

            CM_DLL_API FLOAT GetMaxAxis(UCHAR axis) CONST;

            CM_DLL_API FLOAT GetX(UCHAR minMax) CONST;

            CM_DLL_API FLOAT GetY(UCHAR minMax) CONST;

            CM_DLL_API FLOAT GetZ(UCHAR minMax) CONST;

            CM_DLL_API FLOAT GetRadius(VOID) CONST;

            CM_DLL_API CONST util::Vec3& GetMiddle(VOID) CONST;

            CM_DLL_API CONST Line& GetLine(UCHAR pos) CONST;

            CM_DLL_API BOOL IsInside(CONST util::Vec3& pos) CONST;

            CM_DLL_API VOID SetTransform(util::Mat4& mat);

            CM_DLL_API CONST util::Vec3& GetPoint(UCHAR i) CONST;

            CM_DLL_API ~AxisAlignedBB(VOID);
        };
    }
}