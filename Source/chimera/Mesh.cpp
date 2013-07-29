#include "Mesh.h"

namespace tbd {

    VOID Mesh::AddIndexBufferInterval(UINT start, UINT count, UINT material)
    {
        IndexBufferInterval bi;
        bi.start = start;
        bi.count = count;
        bi.material = material;
        m_indexIntervals.push_back(bi);
    }
}
