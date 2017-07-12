#ifndef UKF_H
#define UKF_H
#include "Eigen/Dense"
#include "measurement_package.h"
#include "ground_truth_package.h"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
  ///*
  ///* "TUNEABLE" CONSTANTS:
  ///*
  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_ = 30;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_ = 30;

  ///* Parameters used to compute lambda_
  const double alpha_ = 0.001; // 1e-3;
  const double beta_ = 2;
  const double k_ = 0;

  ///* if this is false, laser measurements will be ignored (except for init)
  const bool use_laser_ = true;

  ///* if this is false, radar measurements will be ignored (except for init)
  const bool use_radar_ = true;
  
  ///*
  ///*    CONSTANTS
  ///*
  ///* SENSOR CONSTANTS:
  ///* Laser measurement noise standard deviation position1 in m
  const double std_laspx_ = 0.15;

  ///* Laser measurement noise standard deviation position2 in m
  const double std_laspy_ = 0.15;

  ///* Radar measurement noise standard deviation radius in m
  const double std_radr_ = 0.3;

  ///* Radar measurement noise standard deviation angle in rad
  const double std_radphi_ = 0.03;

  ///* Radar measurement noise standard deviation radius change in m/s
  const double std_radrd_ = 0.3;

  ///* MATRIX DIMENSIONS:
  ///* State dimension
  const int n_x_ = 5;

  ///* Augmented state dimension
  const int n_aug_ = 7;


  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* Augmented state vector [pos1 pos2 vel_abs yaw_angle yaw_rate noise-v_accel noise-yaw_accel]
  VectorXd x_aug_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* Augmented covariance matrix
  MatrixXd P_aug_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long time_us_;

  ///* Weights of sigma points for state and covariance
  VectorXd weights_s_; 
  VectorXd weights_c_; 

  ///* Sigma point spreading parameter
  double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   * @param gt_package The ground truth of the state x at measurement time
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

private:
  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  /**
  * Initialize the required variables of the model using the first measurement
   * @param meas_package The measurement at k+1
  */
  void InitializeModel(MeasurementPackage meas_package);

  /**
  * Computes augmented matrices for state (x_aug_) and covariance (P_aug_)
  * from the state (x_) and covariance (P_) matrices.
  */
  void Augmentation();

  /**
  * Generates the augmented sigma points matrix (Xsig) from augmented state 
  * matrix (x_aug_) and augmented covariance matrix (P_aug_)
  */
  MatrixXd GetAugmentedSigmaPoints();

  /**
  * Propagates the motion function. Uses current [px, py, v, yaw, yawd, nu_a, nu_yawdd] to
  * predict a next [px, py, v, yaw, yawd]
  * @Params:  point [px, py, v, yaw, yawd, nu_a, nu_yawdd]
              delta_t is the time between k and k+1 in seconds
  */
  VectorXd FunctionPropagation(VectorXd point, double delta_t);

  /**
  * Normalizes an angle.
  * @Params: angle to normalize in rad
  *          verbose determines if the method displays information  
  */
  double AngleNormalization(double angle, bool verbose = false);

  /**
  * Computes NIS
  * @Params:  z_diff is the difference matrix between the predicted z and the
              measured z
              S is the covariance matrix
  */
  float NIS(VectorXd z_diff, MatrixXd S);
};

#endif /* UKF_H */
