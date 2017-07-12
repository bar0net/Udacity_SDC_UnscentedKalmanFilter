#include <iostream>
#include "ukf.h"

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
  // The model doesn't have initial values
  is_initialized_ = false;

  // Set default value for NIS
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;

  // Set lambda value
  lambda_ = (alpha_*alpha_)*(n_aug_ + k_) - n_aug_;

  // Define the weights (state and covariance)
  double c = n_aug_ + lambda_;
  weights_s_ = (0.5 / c) * VectorXd::Ones(2 * n_aug_ + 1);
  weights_c_ = (0.5 / c) * VectorXd::Ones(2 * n_aug_ + 1);

  weights_s_(0) = (lambda_ / c);
  weights_c_(0) = (1 - (alpha_*alpha_) + beta_) + (lambda_ / c);

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Zero(n_x_, n_x_);
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) return;
  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) return;

  if (!is_initialized_) {
    InitializeModel(meas_package);
    return;
  }
  double v = x_(3);
  double yawd = x_(4);


  double delta_t = (double)(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) UpdateLidar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) UpdateRadar(meas_package);

  // update a and yawdd
  if (delta_t > 0) {
    std_a_ = (x_(3) - v) / delta_t;
    std_yawdd_ = (x_(4) - yawd) / delta_t;
  }
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
  // Update Augmented State and Covariance
  Augmentation();

  // Get the augmented sigma points
  MatrixXd Xsig = GetAugmentedSigmaPoints();

  // Propagate Sigma Points
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    Xsig_pred_.col(i) = FunctionPropagation(Xsig.col(i), delta_t);

  // Use propagated sigma points to predict state
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_s_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = AngleNormalization(x_diff(3)); // Normalize yaw difference

    P_ = P_ + weights_c_(i) * x_diff * x_diff.transpose();
  }

}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = (int)meas_package.raw_measurements_.size();
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // project sigma points to the radar measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig(0, i) = Xsig_pred_(0, i); // px
    Zsig(1, i) = Xsig_pred_(1, i); // py
  }

  // Predicted Measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    z_pred = z_pred + weights_s_(i) * Zsig.col(i);

  // Predicted Measurement Covariance (S) and state-measurement
  // cross-covariance matrix (Tc)
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = AngleNormalization(x_diff(3));

    S = S + weights_c_(i) * z_diff * z_diff.transpose();
    Tc = Tc + weights_c_(i) * x_diff * z_diff.transpose();
  }

  // Add measurement noise to the covariance matrix
  S(0, 0) += std_laspx_*std_laspx_;
  S(1, 1) += std_laspy_*std_laspy_;

  // Kalman Gain
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // Update State and Covariance
  x_ = x_ + K*z_diff;
  P_ = P_ - K * S * K.transpose();
  NIS_laser_ = NIS(z_diff, S);
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = (int)meas_package.raw_measurements_.size();

  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // project sigma points to the radar measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double vx = v * cos(yaw);
    double vy = v * sin(yaw);

    Zsig(0, i) = sqrt(px*px + py*py); // rho
    Zsig(1, i) = atan2(py, px); // theta
    if (Zsig(0, i) > 0.0001)
      Zsig(2, i) = (px*vx + py*vy) / Zsig(0, i);
  }

  // Predicted Measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    z_pred = z_pred + weights_s_(i) * Zsig.col(i);

  // Predicted Measurement Covariance (S) and state-measurement
  // cross-covariance matrix (Tc)
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    z_diff(1) = AngleNormalization(z_diff(1));
    x_diff(3) = AngleNormalization(x_diff(3));

    S = S + weights_c_(i) * z_diff * z_diff.transpose();
    Tc = Tc + weights_c_(i) * x_diff * z_diff.transpose();
  }

  // Add measurement noise to the covariance matrix
  S(0, 0) += std_radr_*std_radr_;
  S(1, 1) += std_radphi_*std_radphi_;
  S(2, 2) += std_radrd_*std_radrd_;

  // Kalman Gain
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = AngleNormalization(z_diff(1));

  // Update State and Covariance
  x_ = x_ + K*z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = NIS(z_diff, S);
}

/**
* Initialize the required variables of the model using the first measurement
* @param meas_package The measurement at k+1
*/
void UKF::InitializeModel(MeasurementPackage meas_package) {
  // get the timestamp
  time_us_ = meas_package.timestamp_;

  // Set the initial state vector
  x_ = VectorXd::Zero(n_x_);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_(0) = meas_package.raw_measurements_[0];  // px
    x_(1) = meas_package.raw_measurements_[1];  // py
    x_(2) = std::sqrt(x_(0)*x_(0) + x_(1)*x_(1)); // ~= v
    x_(3) = std::atan2(x_(1), x_(0)); // ~= theta
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    x_(0) = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]); // px = rho * cos(theta)
    x_(1) = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]); // py = rho * sin(theta)
    x_(2) = meas_package.raw_measurements_[2]; // v ~= rho_dot (Using rho_dot as initial estimation)
  }

  // Set the initial state covariance matrix
  P_ = MatrixXd::Random(n_x_, n_x_);

  // The model is initialized
  is_initialized_ = true;
}

/**
* Computes augmented matrices for state (x_aug_) and covariance (P_aug_)
* from the state (x_) and covariance (P_) matrices.
*/
void UKF::Augmentation() {
  // Set initial augmented state and covariance matrix
  x_aug_ = VectorXd::Zero(n_aug_);
  x_aug_.head(n_x_) = x_;

  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_, n_x_) = std_a_*std_a_;
  P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_* std_yawdd_;

}

/**
* Generates the augmented sigma points matrix (Xsig) from augmented state
* matrix (x_aug_) and augmented covariance matrix (P_aug_)
*/
MatrixXd UKF::GetAugmentedSigmaPoints() {
  MatrixXd Xsig = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  MatrixXd Psqrt = P_aug_.llt().matrixL();
  double c = sqrt(lambda_ + n_aug_);

  Xsig.col(0) = x_aug_;

  for (int i = 0; i < n_aug_; ++i) {
    Xsig.col(1 + i) = x_aug_ + c * Psqrt.col(i);
    Xsig.col(1 + i + n_aug_) = x_aug_ - c * Psqrt.col(i);
  }
  return Xsig;
}

/**
* Propagates the motion function. Uses current [px, py, v, yaw, yawd, nu_a, nu_yawdd] to
* predict a next [px, py, v, yaw, yawd]
* @Params:  point [px, py, v, yaw, yawd, nu_a, nu_yawdd]
delta_t is the time between k and k+1 in seconds
*/
VectorXd UKF::FunctionPropagation(VectorXd point, double delta_t) {
  VectorXd output = VectorXd::Zero(n_x_);

  // Extract falues for readability
  double px = point[0];
  double py = point[1];
  double v = point[2];
  double yaw = point[3];
  double yawd = point[4];
  double nu_a = point[5];
  double nu_yawdd = point[6];

  // Output Values
  double out_px, out_py, out_v, out_yaw, out_yawd;

  // Get next position using v and yaw and avoid division by zero
  if (abs(yawd) > 0.0001) {
    out_px = px + (v / yawd) * (sin(yaw + yawd*delta_t) - sin(yaw));
    out_py = py + (v / yawd) * (cos(yaw) - cos(yaw + yawd*delta_t));
  }
  else {
    out_px = px + v*delta_t*cos(yaw);
    out_py = py + v*delta_t*sin(yaw);
  }

  // Apply nu acceleration to position
  out_px += 0.5 * nu_a * cos(yaw) * delta_t * delta_t;
  out_py += 0.5 * nu_a * sin(yaw) * delta_t * delta_t;

  // compute the rest of the points
  out_v = v + nu_a * delta_t;
  out_yaw = yaw + yawd*delta_t + 0.5*nu_yawdd*delta_t*delta_t;
  out_yawd = yawd + nu_yawdd*delta_t;

  output << out_px, out_py, out_v, out_yaw, out_yawd;

  return output;
}

/**
* Normalizes an angle.
* @Params: angle to normalize in rad
*/
double UKF::AngleNormalization(double angle, bool verbose) {
  //int nloops = (int)std::ceil(angle / (2 * M_PI));
  //std::cout << angle << " -[ " << nloops << " ]->" << (angle - nloops * 2 * M_PI) << std::endl;
  //return (angle - nloops * 2 * M_PI);

  double x = fmod(angle, 2 * M_PI);
  if (x < -M_PI) x += 2 * M_PI;
  else if (x > M_PI) x -= 2 * M_PI;

  //double x = angle - 2.0 * M_PI * floor(angle / (2.0 * M_PI));
  if (verbose) std::cout << angle << " -> " << x << std::endl;

  return x;
}

/**
* Computes NIS
* @Params:  z_diff is the difference matrix between the predicted z and the
measured z
S is the covariance matrix
*/
float UKF::NIS(VectorXd z_diff, MatrixXd S) {
  return (float)(z_diff.transpose() * S.inverse() * z_diff);
}