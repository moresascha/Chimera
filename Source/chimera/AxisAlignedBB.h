#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Mat4.h"
#include "Line.h"

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
public:
    util::Vec3 m_points[8];
    AxisAlignedBB(VOID);
    AxisAlignedBB(CONST AxisAlignedBB& aabb);
    VOID AddPoint(CONST util::Vec3& point);
    VOID Construct(VOID);
    CONST util::Vec3& GetMin(VOID) CONST;
    CONST util::Vec3& GetMax(VOID) CONST;
    FLOAT GetMinAxis(UCHAR axis) CONST;
    FLOAT GetMaxAxis(UCHAR axis) CONST;
    FLOAT GetX(UCHAR minMax) CONST;
    FLOAT GetY(UCHAR minMax) CONST;
    FLOAT GetZ(UCHAR minMax) CONST;
    FLOAT GetRadius(VOID) CONST;
    CONST util::Vec3& GetMiddle(VOID) CONST;
    CONST Line& GetLine(UCHAR pos) CONST;
    BOOL IsInside(CONST util::Vec3& pos) CONST;
    VOID SetTransform(util::Mat4& mat);
    CONST util::Vec3& GetPoint(UCHAR i) CONST;
    ~AxisAlignedBB(VOID);
};
}