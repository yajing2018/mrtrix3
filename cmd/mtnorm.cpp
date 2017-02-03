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

#include <ctime>

#include <limits>
#include "command.h"
#include "image.h"
#include "algo/loop.h"
#include "adapter/extract.h"
#include "filter/optimal_threshold.h"
#include "filter/mask_clean.h"
#include "filter/connected_components.h"
#include "transform.h"
#include "math/least_squares.h"

using namespace MR;
using namespace App;

#define DEFAULT_NORM_VALUE 0.282094
#define DEFAULT_MAXITER_VALUE 20

void usage ()
{
  AUTHOR = "Max";

  DESCRIPTION
   + "Multi-Tissue Intensity Normalisation.";

  ARGUMENTS
    // + Argument ("X", "X").type_file_in();
  + Argument ("input output", "list of all input and output tissue compartment files. See example usage in the description. "
                              "Note that any number of tissues can be normalised").type_image_in().allow_multiple();

  OPTIONS

    + Option ("mask", "define the mask to compute the normalisation within")
    + Argument ("image").type_image_in ()

    + Option ("value", "specify the value to which the summed tissue compartments will be normalised to "
                       "(Default: sqrt(1/(4*pi)) = " + str(DEFAULT_NORM_VALUE, 6) + ")")
    + Argument ("number").type_float ()

    + Option ("robust_iter", "number of iterations")
    + Argument ("number").type_integer ()

    + Option ("ols", "ordinary least squares")
    + Option ("tls", "total least squares");
}

FORCE_INLINE void compute_mask (Image<float>& summed, Image<bool>& mask) {
  LogLevelLatch level (0);
  Filter::OptimalThreshold threshold_filter (summed);
  if (!mask.valid())
    mask = Image<bool>::scratch (threshold_filter);
  threshold_filter (summed, mask);
  Filter::ConnectedComponents connected_filter (mask);
  connected_filter.set_largest_only (true);
  connected_filter (mask, mask);
  Filter::MaskClean clean_filter (mask);
  clean_filter (mask, mask);
}

void run ()
{
  bool do_ols = get_options ("ols").size();
  bool do_tls = get_options ("tls").size();

  if (do_ols + do_tls != 1) {
    throw Exception ("specify one algorithm");
  }

  const float normalisation_value = get_option_value ("value", DEFAULT_NORM_VALUE);
  const size_t robust_iter = get_option_value ("robust_iter", 0);
  if (robust_iter < 0)
    throw Exception ("robust_iter must be nonnegative");

  default_type huber_thresh = 0.5;

  if (argument.size() % 2)
    throw Exception ("The number of input arguments must be even. There must be an output file provided for every input tissue image");

  if (argument.size() < 4)
    throw Exception ("At least two tissue types must be provided");

  ProgressBar progress ("performing intensity normalisation...");
  std::vector<Image<float> > input_images;
  std::vector<Header> output_headers;
  std::vector<std::string> output_filenames;

  // Open input images and check for output
  for (size_t i = 0; i < argument.size(); i += 2) {
    progress++;
    input_images.emplace_back (Image<float>::open (argument[i]));

    // check if all inputs have the same dimensions
    if (i)
      check_dimensions (input_images[0], input_images[i / 2], 0, 3);

    if (Path::exists (argument[i + 1]) && !App::overwrite_files)
      throw Exception ("output file \"" + argument[i + 1] + "\" already exists (use -force option to force overwrite)");

    // we can't create the image yet if we want to put the scale factor into the output header
    output_headers.emplace_back (Header::open (argument[i]));
    output_filenames.push_back (argument[i + 1]);
  }

  Image<bool> mask;
  Header header_3D (input_images[0]);
  header_3D.ndim() = 3;
  auto opt = get_options ("mask");
  if (opt.size()) {
    mask = Image<bool>::open (opt[0][0]);
  } else {
    auto summed = Image<float>::scratch (header_3D);
    for (size_t j = 0; j < input_images.size(); ++j) {
      for (auto i = Loop (summed, 0, 3) (summed, input_images[j]); i; ++i)
        summed.value() += input_images[j].value();
      progress++;
    }
    compute_mask (summed, mask);
  }

  size_t num_voxels = 0;
  for (auto i = Loop (mask) (mask); i; ++i) {
    if (mask.value())
      num_voxels++;
  }

  if (num_voxels == 0)
    throw Exception ("mask empty");

  INFO("num voxels: " + str(num_voxels));

  // Eigen::MatrixXd X = MR::load_matrix<default_type> (argument[0]);

  const size_t num_tissue = input_images.size(); // X.cols();
  VAR(num_tissue);
  // const size_t num_voxels = X.rows();

  Eigen::MatrixXd X (num_voxels, input_images.size());
  size_t index = 0;
  for (auto i = Loop (mask) (mask); i; ++i) {
    if (mask.value()) {
      for (size_t j = 0; j < input_images.size(); ++j) {
        assign_pos_of (mask, 0, 3).to (input_images[j]);
        X (index, j) = input_images[j].value();
      }
      ++index;
    }
  }
  progress++;


  Eigen::MatrixXd scale_factors (num_tissue, 1);
  Eigen::MatrixXd previous_scale_factors (num_tissue, 1);
  Eigen::MatrixXd y (num_voxels, 1);
  y.fill (normalisation_value);

  Eigen::VectorXd rel_residuals (num_voxels);
  std::clock_t start;
  start = std::clock();
  if (do_ols) {
    INFO("OLS");

    // QR:
    // scale_factors = X.colPivHouseholderQr().solve(y);
    // rel_residuals = (X*scale_factors - y);
    // rel_residuals *= 1.0f / normalisation_value;

    scale_factors = X.colPivHouseholderQr().solve(y);

    // Eigen::MatrixXd work(X.cols(),X.cols());
    // Eigen::LLT<Eigen::MatrixXd> llt(work.rows());

    // work.setZero();
    // work.selfadjointView<Eigen::Lower>().rankUpdate (X.transpose());

    // scale_factors = llt.compute (work.selfadjointView<Eigen::Lower>()).solve(X.transpose()*y);

    // if (robust_iter > 0) {
    //   Eigen::VectorXd w (num_voxels);

    //   for (size_t it = 0; it < robust_iter; it++) {
    //     w = Eigen::VectorXd::Ones(num_voxels);
    //     rel_residuals = (X*scale_factors - y);
    //     rel_residuals *= 1.0f / normalisation_value;
    //     for (size_t i = 0; i < num_voxels; i++) {
    //       // if (rel_residuals[i] < (-1.0f * huber_thresh))
    //       if (std::abs(rel_residuals[i]) > huber_thresh)
    //         w(i) = huber_thresh / std::abs(rel_residuals[i]);
    //     }

    //     work.setZero();
    //     work.selfadjointView<Eigen::Lower>().rankUpdate (X.transpose() * w.asDiagonal());

    //     w.array() = w.array().square();
    //     scale_factors = llt.compute (work.selfadjointView<Eigen::Lower>()).solve((X.transpose()*w.asDiagonal()*y));
    //     DEBUG ("               " + str(scale_factors.transpose()));
    //   }
    // }

  } else if (do_tls) {
    INFO("TLS");

    const default_type epsilon (std::numeric_limits<float>::epsilon());
    const default_type data_weighting (1.0e3 / normalisation_value);

    Eigen::MatrixXd A (num_voxels, num_tissue + 1);
    A.block(0,0,num_voxels, num_tissue) = X.block(0,0,num_voxels, num_tissue);
    A.col(num_tissue) = -data_weighting * y;

    // check rank of A
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A);
    lu_decomp.setThreshold(1.0e-9); // num_tissue * epsilon
    size_t rank = lu_decomp.rank();
    if (rank != num_tissue + 1) {
      MAT(X.block(0,0,X.rows() > 100 ? 100 : X.rows(),X.cols()));
      WARN("input data collinear (rank:"+str(rank)+")");
      throw Exception("rank deficient TLS not implemented");
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
    DEBUG("singular values: " + str(svd.singularValues().transpose()));
    // right singular vector corresponding to smallest eigenvector:
    Eigen::VectorXd rsv = svd.matrixV().col(num_tissue);
    DEBUG("right singular vector: " + str(rsv.transpose()));

    // check existence of solution
    if (std::abs(rsv(num_tissue)) < epsilon)
      throw Exception ("no TLS solution exists");
    if (std::abs(svd.singularValues()(num_tissue-1) - svd.singularValues()(num_tissue-2)) < epsilon)
      WARN ("TLS solution not unique");
    rsv.array() /= rsv(num_tissue);
    scale_factors = rsv.head(num_tissue) / data_weighting;

    // robust: repeat until convergence of scale_factors
    //   remove 10% of data with highest equation errors / residuals in y (top and bottom)
    //   reestimate scale_factors
    if (robust_iter > 0) {
      size_t num_remove = std::floor(num_voxels* 0.3 / robust_iter);
      for (size_t it = 0; it < robust_iter; it++) {

        for (size_t bad = 0; it < num_remove; it++)
          A.row(bad).fill(0.0);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
        DEBUG("singular values: " + str(svd.singularValues().transpose()));
        // right singular vector corresponding to smallest eigenvector:
        Eigen::VectorXd rsv = svd.matrixV().col(num_tissue);
        DEBUG("right singular vector: " + str(rsv.transpose()));
      }
    }
  }
  std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  VEC(scale_factors);

  default_type scale_factors_mean = 0.0;
  for (ssize_t i = 0; i < scale_factors.size(); ++i)
    scale_factors_mean += std::log(scale_factors(i, 0));
  scale_factors_mean /= scale_factors.size();
  scale_factors_mean = std::exp (scale_factors_mean);
  VAR(scale_factors_mean);

  default_type residual_y = (X*scale_factors - y).mean()/normalisation_value;
  scale_factors.fill(scale_factors_mean);
  default_type residual_y_mean_scale = (X*scale_factors - y).mean()/normalisation_value;
  VAR(residual_y);
  VAR(residual_y_mean_scale);

  for (size_t j = 0; j < output_filenames.size(); ++j) {
    output_headers[j].keyval()["mtnorm_scale_factor"] = str(scale_factors(j, 0));
    auto output_image = Image<float>::create (output_filenames[j], output_headers[j]);
    for (auto i = Loop (output_image) (output_image, input_images[j]); i; ++i) {
      output_image.value() = scale_factors(j, 0) * input_images[j].value();
    }
  }

}
