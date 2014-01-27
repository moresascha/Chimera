#pragma once
//#include "stdafx.h"
#include <DirectXMath.h>
#include "Vec3.h"
#include "Vec4.h"
namespace chimera
{
    namespace util 
    {
        //using namespace DirectX;
        class Mat4
        {
        private:
            Vec4 m_rotation;
            Vec3 m_translation;
            Vec3 m_scale;
            Vec3 m_pyr;
            XMFLOAT4X4 m_rotationMatrix;
        public:
            XMFLOAT4X4 m_m;

            CM_INLINE Mat4(void);

            CM_INLINE Mat4(const Mat4& mat) 
            {
                memcpy(&this->m_m, &mat.m_m , sizeof(mat.m_m));
                m_rotation = mat.GetRotation();
                m_scale = mat.GetScale();
                m_translation = mat.GetTranslation();
            }

            CM_INLINE Mat4(const XMFLOAT4X4 v)
            {
                m_m = v;
            }

            CM_INLINE Vec3 GetPYR(void) const;

            CM_INLINE void RotateX(float deltaAngle);

            CM_INLINE void RotateY(float deltaAngle);

            CM_INLINE void RotateZ(float deltaAngle);

            CM_INLINE void Rotate(const util::Vec3& axis, float angel);

            CM_INLINE void SetRotateX(float angle);

            CM_INLINE void SetRotateY(float angle);

            CM_INLINE void SetRotateZ(float angle);

            CM_INLINE void SetRotation(const util::Vec3& axis, float angel);

            CM_INLINE void RotateQuat(const util::Vec4& quat);

            CM_INLINE void SetRotateQuat(const util::Vec4& quat);

            CM_INLINE void SetRotateQuat(float x, float y, float z, float w);

            CM_INLINE void SetTranslation(float x, float y, float z) 
            {
                m_translation.Set(x, y, z);
                Update();
                //this->m_m = GetFromMat4x4(XMMatrixTranslation(x, y, z));
            }

            CM_INLINE void SetTranslation(const util::Vec3& pos) 
            {
                SetTranslation(pos.x, pos.y, pos.z);
            }

            CM_INLINE void Translate(float x, float y, float z) 
            {
                m_translation.Add(x, y, z);
                Update();
            }

            CM_INLINE void Translate(const util::Vec3& t) 
            {
                Translate(t.x, t.y, t.z);
            }

            CM_INLINE void Scale(float scale)
            {
                SetScale(GetScale().x * scale, GetScale().z * scale, GetScale().y * scale);
            }

            CM_INLINE void Scale(const util::Vec3& s)
            {
                SetScale(GetScale().x * s.x, GetScale().z * s.y, GetScale().y * s.z);
            }

            CM_INLINE void SetScale(float scale) {
                SetScale(scale, scale, scale);
                //this->m_m = GetFromMat4x4(XMMatrixScaling(x, y ,z));
            }

            CM_INLINE void SetScale(const util::Vec3& scale)
            {
                SetScale(scale.x, scale.y, scale.z);
            }

            CM_INLINE void SetScale(float sx, float sy, float sz)
            {
                this->m_scale.Set(sx, sy, sz);
                Update();
            }

            CM_INLINE const Vec3& GetTranslation(void) const 
            {
                return m_translation;
            }

            CM_INLINE const Vec4& GetRotation(void) const 
            {
                return m_rotation;
            }

            CM_INLINE const util::Vec3& GetScale(void) const 
            {
                return m_scale;
            }

            CM_INLINE util::Vec3 GetPhiTheta(void) const;

            CM_INLINE void Update(void);

            CM_INLINE void Print() const 
            {
                char buff[2048];
                sprintf_s(buff, "\n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n",
                    m_m._11, m_m._21, m_m._31, m_m._41,
                    m_m._12, m_m._22, m_m._32, m_m._42,
                    m_m._13, m_m._23, m_m._33, m_m._43,
                    m_m._14, m_m._24, m_m._34, m_m._44);
                OutputDebugStringA(buff);
            }

            CM_INLINE Mat4& operator= (const Mat4& mat)
            {
                if(this == &mat) return *this;
                memcpy(&this->m_m, &mat.m_m , sizeof(mat.m_m));
                m_rotation = mat.GetRotation();
                m_scale = mat.GetScale();
                m_translation = mat.GetTranslation();
                return *this;
            }

            CM_INLINE ~Mat4(void) {}

            //static stuff

            CM_INLINE static XMFLOAT4X4 GetFromMat4x4(const XMMATRIX& v)
            {
                XMFLOAT4X4 _v;
                XMStoreFloat4x4(&_v, v);
                return _v;
            }

            CM_INLINE static XMMATRIX GetFromFloat4x4(const XMFLOAT4X4& v)
            {
                return XMLoadFloat4x4(&v);
            }

            CM_INLINE static util::Mat4 createLookAtLH(util::Vec3 eyePos, util::Vec3 viewDir, util::Vec3 up) 
            {
                util::Mat4 mat;
                util::Vec3 focusPoint = eyePos + viewDir;
                mat.m_m = GetFromMat4x4(XMMatrixLookAtLH(XMLoadFloat3(&eyePos.m_v), XMLoadFloat3(&focusPoint.m_v), XMLoadFloat3(&up.m_v)));
                return mat;
            }

            CM_INLINE static util::Vec3 Transform(const util::Mat4& mat, const util::Vec3& vec)
            {
                util::Vec4 tmp(vec.x, vec.y, vec.z, 1);
                XMVECTOR v = XMLoadFloat4(&tmp.m_v);
                XMMATRIX m = util::Mat4::GetFromFloat4x4(mat.m_m);
                XMVECTOR res = XMVector4Transform(v, m);
                util::Vec4 _res = util::Vec4::GetFromVector4(res);
                _res.Homogenize();
                return util::Vec3(_res.x, _res.y, _res.z);
            }

            CM_INLINE static util::Vec4 Transform(const util::Mat4& mat, const util::Vec4& vec)
            {
                XMVECTOR v = XMLoadFloat4(&vec.m_v);
                XMMATRIX m = util::Mat4::GetFromFloat4x4(mat.m_m);
                XMVECTOR res = XMVector4Transform(v, m);
                util::Vec4 _res = util::Vec4::GetFromVector4(res);
                return _res;
            }

            CM_INLINE static util::Mat4 Mul(const util::Mat4& left, const util::Mat4& right)
            {
                XMMATRIX _m0 = util::Mat4::GetFromFloat4x4(left.m_m);
                XMMATRIX _m1 = util::Mat4::GetFromFloat4x4(right.m_m);
                XMMATRIX _res = XMMatrixMultiply(_m1, _m0);
                util::Mat4 res = util::Mat4::GetFromMat4x4(_res);
                return res;
            }

            CM_INLINE static void Mul(const util::Mat4& left, const util::Mat4& right, util::Mat4& dst)
            {
                XMMATRIX _m0 = util::Mat4::GetFromFloat4x4(left.m_m);
                XMMATRIX _m1 = util::Mat4::GetFromFloat4x4(right.m_m);
                XMMATRIX _res = XMMatrixMultiply(_m1, _m0);
                dst.m_m = util::Mat4::GetFromMat4x4(_res);
            }

            CM_INLINE static util::Mat4 CreatePerspectiveLH(float fov, float aspect, float zNear, float zFar)
            {
                util::Mat4 projection;
                float invTan = 1.0f / tan(fov * 0.5f);

                projection.m_m._11 = invTan;
                projection.m_m._12 = 0;
                projection.m_m._13 = 0;
                projection.m_m._14 = 0;

                projection.m_m._21 = 0;
                projection.m_m._22 = invTan * aspect;
                projection.m_m._23 = 0;
                projection.m_m._24 = 0;

                projection.m_m._31 = 0;
                projection.m_m._32 = 0;
                projection.m_m._33 = (zFar + zNear) / (zFar - zNear);
                projection.m_m._34 = 1;

                projection.m_m._41 = 0;
                projection.m_m._42 = 0;
                projection.m_m._43 = zFar  * (1 - projection.m_m._33);
                projection.m_m._44 = 0;

                return projection;
            }
        };

        Mat4::Mat4(void)
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

        void Mat4::SetRotateQuat(const util::Vec4& quat)
        {
            SetRotateQuat(quat.x, quat.y, quat.z, quat.w);
        }

        void Mat4::SetRotateQuat(float x, float y, float z, float w)
        {
            m_rotation.m_v.x = x;
            m_rotation.m_v.y = y;
            m_rotation.m_v.z = z;
            m_rotation.m_v.w = w;
            Update();
        }

        Vec3 Mat4::GetPYR(void) const
        {
            float roll = 0;
            float pitch = 0;
            float yaw = 0;
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
                float sqx = m_rotation.x * m_rotation.x;
                float sqy = m_rotation.y * m_rotation.y;
                float sqz = m_rotation.z * m_rotation.z;
                float sqw = m_rotation.w * m_rotation.w;
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

        util::Vec3 Mat4::GetPhiTheta(void) const
        {
            util::Vec4 t = m_rotation;
            t.w = 0;
            if(t.Length() > 0)
            {
                util::Vec3 pt;
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

        void Mat4::RotateQuat(const util::Vec4& quat)
        {
            XMVECTOR q0 = XMLoadFloat4(&quat.m_v);
            XMVECTOR q1 = XMLoadFloat4(&m_rotation.m_v);

            XMVECTOR result = XMQuaternionMultiply(q0, q1);

            XMFLOAT4 sq;
            XMStoreFloat4(&sq, result);

            SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
        }

        void Mat4::Rotate(const util::Vec3& axis, float angel) 
        {
            XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
            XMVECTOR result = XMQuaternionMultiply(XMLoadFloat4(&m_rotation.m_v), q);
            XMFLOAT4 sq;
            XMStoreFloat4(&sq, result);
            SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
        }

        void Mat4::SetRotation(const util::Vec3& axis, float angel)
        {
            XMVECTOR q = XMQuaternionRotationNormal(XMLoadFloat3(&axis.m_v), angel);
            XMFLOAT4 sq;
            XMStoreFloat4(&sq, q);
            SetRotateQuat(sq.x, sq.y, sq.z, sq.w);
        }

        void Mat4::RotateX(float deltaAngle)
        {
            static Vec3 X_AXIS(1,0,0);
            Rotate(X_AXIS, deltaAngle);
        }

        void Mat4::RotateY(float deltaAngle)
        {
            static Vec3 Y_AXIS(0,1,0);
            Rotate(Y_AXIS, deltaAngle);
        }

        void Mat4::RotateZ(float deltaAngle)
        {
            static Vec3 Z_AXIS(0,0,1);
            Rotate(Z_AXIS, deltaAngle);
        }

        void Mat4::SetRotateX(float angle)
        {
            static Vec3 X_AXIS(1,0,0);
            SetRotation(X_AXIS, angle);
        }

        void Mat4::SetRotateY(float angle)
        {
            static Vec3 Y_AXIS(0,1,0);
            SetRotation(Y_AXIS, angle);
        }

        void Mat4::SetRotateZ(float angle)
        {
            static Vec3 Z_AXIS(0,0,1);
            SetRotation(Z_AXIS, angle);
        }

        void Mat4::Update(void)
        {
            XMStoreFloat4x4(
                &m_m, 
                XMMatrixScaling(m_scale.x, m_scale.y, m_scale.z) * XMMatrixRotationQuaternion(XMLoadFloat4(&m_rotation.m_v)) *
                XMMatrixTranslation(m_translation.x, m_translation.y, m_translation.z)
                );
        }
    }
}
