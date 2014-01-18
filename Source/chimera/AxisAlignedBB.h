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
            float m_radius;
            util::Vec3 m_points[8];

        public:
            CM_DLL_API AxisAlignedBB(void);

            CM_DLL_API AxisAlignedBB(const AxisAlignedBB& aabb);

            CM_DLL_API void Clear(void);

            CM_DLL_API void AddPoint(const util::Vec3& point);

            CM_DLL_API void Construct(void);

            CM_DLL_API const util::Vec3& GetMin(void) const;

            CM_DLL_API const util::Vec3& GetMax(void) const;

            CM_DLL_API float GetMinAxis(UCHAR axis) const;

            CM_DLL_API float GetMaxAxis(UCHAR axis) const;

            CM_DLL_API float GetX(UCHAR minMax) const;

            CM_DLL_API float GetY(UCHAR minMax) const;

            CM_DLL_API float GetZ(UCHAR minMax) const;

            CM_DLL_API float GetRadius(void) const;

            CM_DLL_API const util::Vec3& GetMiddle(void) const;

            CM_DLL_API const Line& GetLine(UCHAR pos) const;

            CM_DLL_API bool IsInside(const util::Vec3& pos) const;

            CM_DLL_API void SetTransform(util::Mat4& mat);

            CM_DLL_API const util::Vec3& GetPoint(UCHAR i) const;

            CM_DLL_API ~AxisAlignedBB(void);
        };
    }
}