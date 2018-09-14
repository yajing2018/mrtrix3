/*
 * Copyright (c) 2008-2018 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/
 *
 * MRtrix3 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For more details, see http://www.mrtrix.org/
 */


#include "command.h"
#include "image.h"
#include "phase_encoding.h"
#include "progressbar.h"
#include "math/SH.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "algo/threaded_loop.h"

#include <Eigen/SVD>

using namespace MR;
using namespace App;

void usage ()
{
  AUTHOR = "J-Donald Tournier (jdtournier@gmail.com)";

  SYNOPSIS = "Compute the SVD of a 4D image, by arranging the data as a matrix of voxels by volumes";

  DESCRIPTION
  + "TBC";

  ARGUMENTS
  + Argument ("input", "the input image.").type_image_in ()
  + Argument ("weights", "the output weights image.").type_image_out ()
  + Argument ("values", "the text file into which the output singular values are to be stored").type_file_out()
  + Argument ("vectors", "the text file into which the output right singular vectors are to be stored").type_file_out();



  OPTIONS
  + Option ("mask",
            "only use voxels within the mask image provided.")
  +   Argument ("noise").type_image_in();
}





using value_type = double;


void run ()
{
  auto in = Image<value_type>::open (argument[0]);

  if (in.ndim() != 4)
    throw Exception ("input data must be 4D");

  size_t count = voxel_count (in, 0, 3);
  Image<bool> mask;

  auto opt = get_options ("mask");
  if (opt.size()) {
    count = 0;
    mask = Image<bool>::open (opt[0][0]);
    for (auto l = Loop(0,3)(mask); l; ++l)
      if (mask.value())
        ++count;
    INFO ("found " + str(count) + " voxels in mask");
  }

  Eigen::MatrixXd data (count, in.size(3));

  Header header (in);
  header.datatype() = DataType::Float32;
  auto out = Image<value_type>::create (argument[1], header);

  size_t n = 0;
  for (auto l = Loop("loading input image \"" + in.name() + "\"",0,3)(in); l; ++l) {
    if (mask.valid()) {
      assign_pos_of (in, 0, 3).to (mask);
      if (!mask.value())
        continue;
    }

    for (auto l2 = Loop(3)(in); l2; ++l2)
      data(n,in.index(3)) = in.value();
    ++n;
  }

  Timer timer;
  CONSOLE ("computing SVD...");
  Eigen::BDCSVD<Eigen::MatrixXd> svd (data, Eigen::ComputeThinU | Eigen::ComputeThinV);
  INFO ("SVD computed in " + str(timer.elapsed()) + " seconds");

  save_matrix (svd.singularValues(), argument[2]);
  save_matrix (svd.matrixV(), argument[3]);

  n = 0;
  for (auto l = Loop("storing output image \"" + out.name() + "\"",0,3)(out); l; ++l) {
    if (mask.valid()) {
      assign_pos_of (out, 0, 3).to (mask);
      if (!mask.value())
        continue;
    }

    for (auto l2 = Loop(3)(out); l2; ++l2)
      out.value() = svd.matrixU()(n, out.index(3));
    ++n;
  }
}

