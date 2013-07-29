#pragma once
#include "Material.h"
namespace tbd {
    class IRenderer {
    protected:
        util::Vec4 m_backgroundColor;
    public:
        virtual VOID VSetBackground(CHAR r, CHAR g, CHAR b, CHAR a) {
            m_backgroundColor.Set(r, g, b, a);
        }
        virtual UINT VGetWidth(VOID) = 0;
        virtual UINT VGetHeight(VOID) = 0;
        virtual HRESULT VOnRestore(VOID) = 0;
        virtual VOID VPreRender(VOID) = 0;
        virtual VOID VPostRender(VOID) = 0;
        virtual VOID VPresent(VOID) = 0;

        virtual VOID VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos) = 0;
        virtual VOID VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance) = 0;
        virtual VOID VSetWorldTransform(CONST util::Mat4& mat) = 0;

        virtual VOID VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos) = 0;
        virtual VOID VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance) = 0;
        virtual VOID VPushWorldTransform(CONST util::Mat4& mat) = 0;

        virtual VOID VPopViewTransform(VOID) = 0;
        virtual VOID VPopProjectionTransform(VOID) = 0;
        virtual VOID VPopWorldTransform(VOID) = 0;

        virtual VOID VPushAmbientColor(util::Vec3& aColor) {}
        virtual VOID VPushSpecularColor(util::Vec3& aColor) {}
        virtual VOID VPushDiffuseColor(util::Vec3& aColor) {}
        virtual VOID VPushSpecularExp(FLOAT expo) {}
        virtual VOID VPushMaterial(tbd::IMaterial& mat) {}
        virtual VOID VPushPrimitiveType(UINT type) = 0;

        virtual VOID VPopAmbientColor(VOID) {}
        virtual VOID VPopSpecularColor(VOID) {}
        virtual VOID VPopDiffuseColor(VOID) {}
        virtual VOID VPopSpecularExp(VOID) {}
        virtual VOID VPopMaterial(VOID) {}
        virtual VOID VPopPrimitiveType(VOID) {}
        virtual ~IRenderer() { }
    };
}

