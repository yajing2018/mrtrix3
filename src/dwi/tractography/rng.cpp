#include "dwi/tractography/rng.h"

namespace MR
{
  namespace DWI
  {
    namespace Tractography
    {

      Math::RNG& rng () {
        static thread_local Math::RNG thread_local_rng;
        return thread_local_rng;
      }

    }
  }
}

