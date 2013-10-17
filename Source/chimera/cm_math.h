#pragma once

namespace chimera
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
