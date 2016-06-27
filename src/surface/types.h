/*
 * Copyright (c) 2008-2016 the MRtrix3 contributors
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/
 * 
 * MRtrix is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * For more details, see www.mrtrix.org
 * 
 */

#ifndef __surface_types_h__
#define __surface_types_h__



#include "surface/polygon.h"



namespace MR
{
  namespace Surface
  {


    typedef Eigen::Vector3 Vertex;
    typedef std::vector<Vertex> VertexList;
    typedef Polygon<3> Triangle;
    typedef Polygon<4> Quad;
    typedef std::vector<Triangle> TriangleList;
    typedef std::vector<Quad> QuadList;

    class Vox : public Eigen::Array3i
    {
      public:
        using Eigen::Array3i::Array3i;
        bool operator< (const Vox& i) const
        {
          return ((*this)[2] == i[2] ? (((*this)[1] == i[1]) ? ((*this)[0] < i[0]) : ((*this)[1] < i[1])) : ((*this)[2] < i[2]));
        }
    };



  }
}

#endif
