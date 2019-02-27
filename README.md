# Orientation-Tracking-Using-Unscented-Kalman-Filter

In this Project, I have implemented an Unscented Kalman Filter to track three dimensional orientation. This means to estimate the underlying 3D orientation by learning the appropriate model parameters from ground truth data given by a Vicon motion capture system, given IMU sensor readings from gyroscopes and accelerometers.

* Tested on: Ubuntu 18.04, Intel® Core™ i7-7500U CPU @ 2.70GHz × 4 
* Python 3.6.7 |Anaconda custom (64-bit)|

### Challenge Description

* Calculate Bias and Scale parameters for accelerometer and gyroscope readings.
* Convert IMU Readings to Quaternions.
* Implement Unscented Kalman Filters.

### Unscented Kalman Filter

#### Process Model

* The UKF implementation was done using only orientation(gyroscope) in the state vector as the control input: q = [q0, q1, q2, q3]^T
* Initialize P (Covariance matrix) as size of 3x3. Similarly, R and Q. R is measurement noise and Q is process noise.
* After Kalman filter predict step, new P and state vector q are obtained, which are the used for update step.
* Then Sigma Points are obtained by Cholesky decomposition of (P+Q).

#### Motion Model

* This step deals with updating P and getting new mean state q. Which then leads to obtaining new Sigma Points. This new sigma points are used to calculate multiple covariances, like Pzz, Pxz, and Pvv.
* The next step involves computing K (Kalman Gain) = Pxz Pvv-1 and I (Innovation term) = Accelerometer reading – Mean of Sigma Points
* These are used to calculate the P and q for the next stage.



### Results

![](https://github.com/Iron-Stark/Orientation-Tracking-Using-Unscented-Kalman-Filter/blob/master/results/IMU_1.png)

![](https://github.com/Iron-Stark/Orientation-Tracking-Using-Unscented-Kalman-Filter/blob/master/results/IMU_2.png)

![](https://github.com/Iron-Stark/Orientation-Tracking-Using-Unscented-Kalman-Filter/blob/master/results/IMU_3.png)

### Instructions

Run python3 estimate_rot.py to visualize the results.



### References

[]: https://ieeexplore.ieee.org/document/1257247	"A Quaternion Based UKF for orientation tracking"

