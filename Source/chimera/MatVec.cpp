#include "stdafx.h"
#include "Mat4.h"
#include "Vec3.h"
#include "Vec4.h"
#include "math.h"

namespace util 
{
    CONST Mat4 Mat4::IDENTITY = Mat4();

    CONST Vec3 Vec3::X_AXIS = Vec3(1,0,0);
    CONST Vec3 Vec3::Y_AXIS = Vec3(0,1,0);
    CONST Vec3 Vec3::Z_AXIS = Vec3(0,0,1);

    Mat4::Mat4(VOID)
    {
        m_scale.Set(1, 1, 1);

        m_m._11 = 1;
        m_m._12 = 0;
        m_m._13 = 0;
        m_m._14 = 0;

        m_m._21 = 0;
        m_m._22 = 1;
        m_m._23 = 0;
        m_m._24 = 0;

        m_m._31 = 0;
        m_m._32 = 0;
        m_m._33 = 1;
        m_m._34 = 0;

        m_m._41 = 0;
        m_m._42 = 0;
        m_m._43 = 0;
        m_m._44 = 1; 

        m_rotation.w = -1;
    }

    VOID Mat4::SetRotateQuat(CONST util::Vec4& quat)
    {
        SetRotateQuat(quat.x, quat.y, quat.z, quat.w);
    }

    VOID Mat4::SetRotateQuat(FLOAT x, FLOAT y, FLOAT z, FLOAT w)
    {
        m_rotation.m_v.x = x;
        m_rotation.m_v.y = y;
        m_rotation.m_v.z = z;
        m_rotation.m_v.w = w;
        Update();
    }

    Vec3 Mat4::GetPYR(VOID) CONST
    {
        FLOAT roll = 0;
        FLOAT pitch = 0;
        FLOAT yaw = 0;
        DOUBLE test = m_rotation.x * m_rotation.y + m_rotation.z * m_rotation.w;

        /*
        if (test > 0.499) 
        { // singularity at north pole
            heading = 2 * atan2(m_rotation.y, m_rotation.x);
            attitude = XM_PIDIV2;
            bank = 0;
        } 
        else if (test < -0.499) 
        { // singularity at south pole
            heading = -2 * atan2(m_rotation.y, m_rotation.x);
            attitude = -XM_PI/2;
            bank = 0;
        } 
        else */
        {
            FLOAT sqx = m_rotation.x * m_rotation.x;
            FLOAT sqy = m_rotation.y * m_rotation.y;
            FLOAT sqz = m_rotation.z * m_rotation.z;
            FLOAT sqw = m_rotation.w * m_rotation.w;
            roll  = atan2(2 * m_rotation.x * m_rotation.y + 2 * m_rotation.z * m_rotation.w, 1 - 2 * sqy - 2 * sqz);
            pitch = asin(2 * m_rotation.x * m_rotation.z - 2 * m_rotation.w * m_rotation.y);
            yaw   = atan2(2 * m_rotation.x * m_rotation.w + 2 * m_rotation.y * m_rotation.z, 1 - 2 * sqz - 2 * sqw);
        }
        util::Vec3 pyr;
        pyr.z = roll;
        pyr.y = pitch;
        pyr.x = -yaw;
        return pyr;
    }

    util::Vec3 Mat4::GetPhiTheta(VOID) CONST
    {
        util::Vec4 t = m_rotation;
        t.w = 0;
        if(t.Length() > 0)
        {
            util::Vec3 pt;
            XMVECTOR v;
            FLOAT angle;
            //XMQuaternionToAxisAngle(&v, &angle, XMLoadFloat4(&m_rotation.m_v));
            //v.m128_f32[1] = 0;
            util::Vec3 up(0,1,0);
            up = Mat4::Transform(*this, up);
            up.Normalize();
            DEBUG_OUT("---\n");
            
            up.Print();
            pt.x = 2 * acos(up.y);
            DEBUG_OUT_A("%f\n", RAD_TO_DEGREE(pt.x));
            /*util::Vec3 r(1,0,0);
            r = Mat4::Transform(*this, r);
            r.Normalize();
            v = XMLoadFloat3(&(r.m_v));
            xa = XMVector3AngleBetweenVectors(v, XMLoadFloat3(&Vec3::X_AXIS.m_v));
            pt.y = 2 * xa.m128_f32[0]; */
            pt.Print();
            return pt;
        }
        else
        {
            return util::Vec3();
        }
    }

    VOID Mat4::RotateQuat(CONST util::Vec4& quat)
    {
        XMVECTOR q0 = XMLoadFloat4(&quat.m_v);
        XMVECTOR q1 = XMLoadFloat4(&m_rotation.m_v);

        XMVECTOR result = XMQuaternionMultiply(q0, q1);

        XMFLOAT4 sq;
        XMStoreFloat4(&sq, result);

        SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
    }

    VOID Mat4::Rotate(CONST util::Vec3& axis, FLOAT angel) 
    {
        XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
        XMVECTOR result = XMQuaternionMultiply(XMLoadFloat4(&m_rotation.m_v), q);
        XMFLOAT4 sq;
        XMStoreFloat4(&sq, result);
        SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
    }

    VOID Mat4::SetRotation(CONST util::Vec3& axis, FLOAT angel)
    {
        XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
        XMFLOAT4 sq;
        XMStoreFloat4(&sq, q);
        SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
    }

    VOID Mat4::RotateX(FLOAT deltaAngle)
    {
        Rotate(Vec3::X_AXIS, deltaAngle);
    }

    VOID Mat4::RotateY(FLOAT deltaAngle)
    {
        Rotate(Vec3::Y_AXIS, deltaAngle);
    }

    VOID Mat4::RotateZ(FLOAT deltaAngle)
    {
        Rotate(Vec3::Z_AXIS, deltaAngle);
    }

    VOID Mat4::SetRotateX(FLOAT angle)
    {
        SetRotation(Vec3::X_AXIS, angle);
    }

    VOID Mat4::SetRotateY(FLOAT angle)
    {
        SetRotation(Vec3::Y_AXIS, angle);
    }

    VOID Mat4::SetRotateZ(FLOAT angle)
    {
        SetRotation(Vec3::Z_AXIS, angle);
    }

    VOID Mat4::Update(VOID)
    {
        XMStoreFloat4x4(
            &m_m, 
            XMMatrixScaling(m_scale.x, m_scale.y, m_scale.z) * XMMatrixRotationQuaternion(XMLoadFloat4(&m_rotation.m_v)) *
            XMMatrixTranslation(m_translation.x, m_translation.y, m_translation.z)
            );
    }
}