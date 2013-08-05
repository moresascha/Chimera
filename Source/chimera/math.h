#pragma once

#define CLAMP(x, low, high)  ((x) = (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x))))
#define DEGREE_TO_RAD(__deg_) (FLOAT)(((__deg_) / 180.0 * XM_PI))
#define RAD_TO_DEGREE(__deg_) (FLOAT)(((__deg_) * 180.0 / XM_PI))

namespace tbd
{
    namespace math
    {
        template <class T>
        INT sign(T t)
        {
            return t < (T)0 ? -1 : (t > (T)0 ? 1 : 0);
        }
    }
}
