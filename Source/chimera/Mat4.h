#pragma once
#include "stdafx.h"
#include "Vec3.h"
namespace util 
{
//using namespace DirectX;
    class DLL_EXPORT Mat4
    {
    private:
        Vec4 m_rotation;
        Vec3 m_translation;
        Vec3 m_scale;
        Vec3 m_pyr;
        XMFLOAT4X4 m_rotationMatrix;
    public:
        static CONST Mat4 IDENTITY;

        XMFLOAT4X4 m_m;

        Mat4(VOID);

        Mat4(CONST Mat4& mat) 
        {
            memcpy(&this->m_m, &mat.m_m , sizeof(mat.m_m));
            m_rotation = mat.GetRotation();
            m_scale = mat.GetScale();
            m_translation = mat.GetTranslation();
        }

        Mat4(CONST XMFLOAT4X4 v)
        {
            m_m = v;
        }

        Vec3 GetPYR(VOID) CONST;

        VOID RotateX(FLOAT deltaAngle);

        VOID RotateY(FLOAT deltaAngle);

        VOID RotateZ(FLOAT deltaAngle);

        VOID Rotate(CONST util::Vec3& axis, FLOAT angel);

        VOID SetRotateX(FLOAT angle);

        VOID SetRotateY(FLOAT angle);

        VOID SetRotateZ(FLOAT angle);

        VOID SetRotation(CONST util::Vec3& axis, FLOAT angel);

        VOID RotateQuat(CONST util::Vec4& quat);

        VOID SetRotateQuat(CONST util::Vec4& quat);

        VOID SetRotateQuat(FLOAT x, FLOAT y, FLOAT z, FLOAT w);

        inline VOID SetTranslate(FLOAT x, FLOAT y, FLOAT z) 
        {
            m_translation.Set(x, y, z);
            Update();
            //this->m_m = GetFromMat4x4(XMMatrixTranslation(x, y, z));
        }

        inline VOID SetTranslate(CONST util::Vec3& pos) 
        {
            SetTranslate(pos.x, pos.y, pos.z);
        }

        inline VOID Translate(FLOAT x, FLOAT y, FLOAT z) 
        {
            m_translation.Add(x, y, z);
            Update();
        }

        inline VOID Translate(CONST util::Vec3& t) 
        {
            Translate(t.x, t.y, t.z);
        }

        inline VOID Scale(FLOAT scale)
        {
            SetScale(GetScale().x * scale, GetScale().z * scale, GetScale().y * scale);
        }

        inline VOID Scale(CONST util::Vec3& s)
        {
            SetScale(GetScale().x * s.x, GetScale().z * s.y, GetScale().y * s.z);
        }

        inline VOID SetScale(FLOAT scale) {
            SetScale(scale, scale, scale);
            //this->m_m = GetFromMat4x4(XMMatrixScaling(x, y ,z));
        }

        inline VOID SetScale(FLOAT sx, FLOAT sy, FLOAT sz)
        {
            this->m_scale.Set(sx, sy, sz);
            Update();
        }

        inline CONST Vec3& GetTranslation(VOID) CONST 
        {
            return m_translation;
        }

        inline CONST Vec4& GetRotation(VOID) CONST 
        {
            return m_rotation;
        }

        inline CONST util::Vec3& GetScale(VOID) CONST 
        {
            return m_scale;
        }

        util::Vec3 GetPhiTheta(VOID) CONST;

        VOID Update(VOID);

        inline VOID Print() CONST 
        {
            char buff[2048];
            sprintf_s(buff, "\n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n",
                m_m._11, m_m._21, m_m._31, m_m._41,
                m_m._12, m_m._22, m_m._32, m_m._42,
                m_m._13, m_m._23, m_m._33, m_m._43,
                m_m._14, m_m._24, m_m._34, m_m._44);
            OutputDebugStringA(buff);
        }

        Mat4& operator= (CONST Mat4& mat)
        {
            if(this == &mat) return *this;
            memcpy(&this->m_m, &mat.m_m , sizeof(mat.m_m));
            m_rotation = mat.GetRotation();
            m_scale = mat.GetScale();
            m_translation = mat.GetTranslation();
            return *this;
        }

        ~Mat4(VOID) {}

        //static stuff

        inline static XMFLOAT4X4 GetFromMat4x4(CONST XMMATRIX& v)
        {
            XMFLOAT4X4 _v;
            XMStoreFloat4x4(&_v, v);
            return _v;
        }

        inline static XMMATRIX GetFromFloat4x4(CONST XMFLOAT4X4& v)
        {
            return XMLoadFloat4x4(&v);
        }

        inline static util::Mat4 createLookAtLH(util::Vec3 eyePos, util::Vec3 viewDir, util::Vec3 up) 
        {
            util::Mat4 mat;
            util::Vec3 focusPoint = eyePos + viewDir;
            mat.m_m = GetFromMat4x4(XMMatrixLookAtLH(XMLoadFloat3(&eyePos.m_v), XMLoadFloat3(&focusPoint.m_v), XMLoadFloat3(&up.m_v)));
            return mat;
        }

        inline static util::Vec3 Transform(CONST util::Mat4& mat, CONST util::Vec3& vec)
        {
            util::Vec4 tmp(vec.x, vec.y, vec.z, 1);
            XMVECTOR v = XMLoadFloat4(&tmp.m_v);
            XMMATRIX m = util::Mat4::GetFromFloat4x4(mat.m_m);
            XMVECTOR res = XMVector4Transform(v, m);
            util::Vec4 _res = util::Vec4::GetFromVector4(res);
            _res.Homogenize();
            return util::Vec3(_res.x, _res.y, _res.z);
        }

        inline static util::Vec4 Transform(CONST util::Mat4& mat, CONST util::Vec4& vec)
        {
            XMVECTOR v = XMLoadFloat4(&vec.m_v);
            XMMATRIX m = util::Mat4::GetFromFloat4x4(mat.m_m);
            XMVECTOR res = XMVector4Transform(v, m);
            util::Vec4 _res = util::Vec4::GetFromVector4(res);
            return _res;
        }

        inline static util::Mat4 Mul(CONST util::Mat4& left, CONST util::Mat4& right)
        {
            XMMATRIX _m0 = util::Mat4::GetFromFloat4x4(left.m_m);
            XMMATRIX _m1 = util::Mat4::GetFromFloat4x4(right.m_m);
            XMMATRIX _res = XMMatrixMultiply(_m1, _m0);
            util::Mat4 res = util::Mat4::GetFromMat4x4(_res);
            return res;
        }

        inline static util::Mat4 CreatePerspectiveLH(FLOAT fov, FLOAT aspect, FLOAT zNear, FLOAT zFar)
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
}