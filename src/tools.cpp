#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse = VectorXd::Zero(4);

  // Input data sanity check
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cout << "Error: Invalid input in CalculateRMSE().";
    return rmse;
  }

  VectorXd error;
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    // Get entry error
    error = estimations[i] - ground_truth[i];

    // Square entry error
    error = error.array()*error.array();

    // accumulate each squared entry error
    rmse += error;
  }

  // Calculate mean
  rmse = rmse / estimations.size();

  // Output the square root
  rmse =  rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) { 
	MatrixXd Hj(3, 4);

  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float sum2 = px*px + py*py;
  float root = std::sqrt(sum2);
  float sum2_x_root = sum2 * root;

  // check: division not by zero
  if (std::fabs(sum2) > 0.0001) {
      // Compute Jacobian
      Hj << (px / root), (py / root), 0, 0,
        -(py / sum2), (px / sum2), 0, 0,
        py*(vx*py - vy*px) / sum2_x_root, px*(px*vy - py*vx) / sum2_x_root, px / root, py / root;
  }
  else {
    std::cout << "Error: Division by zero in CalculateJacobian()." << std::endl;
  }

  return Hj;
}
