#pragma once
#include "api/CMDefines.h"
#include "Vec3.h"
#include <DirectXMath.h>

using namespace DirectX;

namespace chimera
{
    namespace util
    {
        class Quat
        {
        public:
            XMFLOAT4 m_v;
            XMFLOAT4 m_o;

            CM_INLINE Quat(void);

            CM_INLINE Quat(const Quat& q);

            CM_INLINE Quat(const Vec4& q);

            CM_INLINE Quat(float x, float y, float z, float w);

            CM_INLINE Quat Inverse(void);

            CM_INLINE void Transform(util::Vec3& v);

            CM_INLINE void Rotate(const util::Vec3& axis, float angel);

            CM_INLINE void SetRotation(const util::Vec3& axis, float angel);

            CM_INLINE void Translate(const util::Vec3& origin);

        };

        Quat::Quat(void) : m_v(0,0,0,1), m_o(0,0,0,0)
        {

        }

        Quat::Quat(const Quat& q)
        {
            m_v = q.m_v;
            m_o = q.m_o;
        }

        Quat::Quat(const Vec4& q)
        {
            m_v = q.m_v;
            m_o.x = m_o.y = m_o.z = m_o.w = 0;
        }

        Quat::Quat(float x, float y, float z, float w)
        {
            m_v.x = x;
            m_v.y = y;
            m_v.z = z;
            m_v.w = w;
            m_o.x = m_o.y = m_o.z = m_o.w = 0;
        }

        void Quat::Transform(util::Vec3& v)
        {
            XMFLOAT3 diff;
            diff.x = v.x - m_o.x;
            diff.y = v.y - m_o.y;
            diff.z = v.z - m_o.z;
            XMVECTOR _tmp = XMVector3Rotate(XMLoadFloat3(&diff), XMLoadFloat4(&m_v));
            XMFLOAT4 dst;
            XMStoreFloat4(&dst, _tmp);
            v.x = dst.x + m_o.x;
            v.y = dst.y + m_o.y;
            v.z = dst.z + m_o.z;
        }

        void Quat::Translate(const util::Vec3& origin)
        {
            m_o.x = origin.m_v.x;
            m_o.y = origin.m_v.y;
            m_o.z = origin.m_v.z;
            m_o.w = 0;
        }

        void Quat::Rotate(const util::Vec3& axis, float angel) 
        {
            XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
            XMVECTOR result = XMQuaternionMultiply(XMLoadFloat4(&m_v), q);
            XMStoreFloat4(&m_v, result);
        }

        void Quat::SetRotation(const util::Vec3& axis, float angel)
        {
            XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
            XMStoreFloat4(&m_v, q);
        }

        Quat Quat::Inverse(void)
        {
            XMVECTOR result = XMQuaternionInverse(XMLoadFloat4(&m_v));
            Quat q;
            XMStoreFloat4(&q.m_v, result);
            return q;
        }
    }
}
