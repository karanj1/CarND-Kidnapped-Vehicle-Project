/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//std::normal-distribution  :  https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution

	num_particles = 101;

  weights.resize(num_particles);
    // define normal distributions for sensor noise
    normal_distribution<double> N_x_init(0, std[0]);
    normal_distribution<double> N_y_init(0, std[1]);
    normal_distribution<double> N_theta_init(0, std[2]);  

    // init particles
    for (int i = 0; i < num_particles; i++) {
      Particle p;
      p.id = i;
      p.x = x;
      p.y = y;
      p.theta = theta;
      p.weight = 1.0;  

      // add noise
      p.x += N_x_init(gen);
      p.y += N_y_init(gen);
      p.theta += N_theta_init(gen);  

      particles.push_back(p);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distributions for sensor noise
    normal_distribution<double> NoiseDist_x(0, std_pos[0]);
    normal_distribution<double> NoiseDist_y(0, std_pos[1]);
    normal_distribution<double> NoiseDist_theta(0, std_pos[2]);  

    for (int i = 0; i < num_particles; i++) {  

      // calculate new state / prediction  //Lesson 12(Motion Model) Session 3  
      if (fabs(yaw_rate) < 0.00001) {  
        particles[i].x += velocity * delta_t * cos(particles[i].theta);
        particles[i].y += velocity * delta_t * sin(particles[i].theta);
      } 
      else {
        particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
        particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
        particles[i].theta += yaw_rate * delta_t;
      }  

      // add noise
      particles[i].x += NoiseDist_x(gen);
      particles[i].y += NoiseDist_y(gen);
      particles[i].theta += NoiseDist_theta(gen);
  }

}

/*
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
    
      // grab current observation
      LandmarkObs o = observations[i];  

      // init minimum distance to maximum possible
      double min_dist = numeric_limits<double>::max();  

      // init id of landmark from map placeholder to be associated with the observation
      int map_id = -1;
      
      for (unsigned int j = 0; j < predicted.size(); j++) {
        // grab current prediction
        LandmarkObs p = predicted[j];
        
        // get distance between current/predicted landmarks
        double cur_dist = dist(o.x, o.y, p.x, p.y);  

        // find the predicted landmark nearest the current observed landmark
        if (cur_dist < min_dist) {
          min_dist = cur_dist;
          map_id = p.id;
        }
      }

      // set the observation's id to the nearest predicted landmark's id
      observations[i].id = map_id;
    }

}
*/

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // First term of multi-variate normal Gaussian distribution calculated below
  // It stays the same so can be outside the loop
  const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  // The denominators of the mvGd also stay the same
  const double x_denom = 2 * std_landmark[0] * std_landmark[0];
  const double y_denom = 2 * std_landmark[1] * std_landmark[1];

  // Iterate through each particle
  for (int i = 0; i < num_particles; ++i) {
    
    // For calculating multi-variate Gaussian distribution of each observation, for each particle
    double mvGd = 1.0;
    
    // For each observation
    for (int j = 0; j < observations.size(); ++j) {
      
      // Transform the observation point (from vehicle coordinates to map coordinates)
      double trans_obs_x, trans_obs_y;
      trans_obs_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      trans_obs_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      
      // Find nearest landmark
      vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
      vector<double> landmark_obs_dist (landmarks.size());
      for (int k = 0; k < landmarks.size(); ++k) {
        
        // Down-size possible amount of landmarks to look at by only looking at those in sensor range of the particle
        // If in range, put in the distance vector for calculating nearest neighbor
        double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));

        if (landmark_part_dist <= sensor_range) {
          landmark_obs_dist[k] = sqrt(pow(trans_obs_x - landmarks[k].x_f, 2) + pow(trans_obs_y - landmarks[k].y_f, 2));

        } else {
          // Need to fill those outside of distance with huge number, or they'll be a zero (and think they are closest)
          landmark_obs_dist[k] = 999999.0;
          
        }
        
      }
      
      // Associate the observation point with its nearest landmark neighbor
      //http://www.cplusplus.com/reference/iterator/distance/
      //finds the 'position' of the Landmark element having minimum distance w.r.t observations[j]
      int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
      float nn_x = landmarks[min_pos].x_f;
      float nn_y = landmarks[min_pos].y_f;
      
      // Calculate multi-variate Gaussian distribution
      double x_diff = trans_obs_x - nn_x;
      double y_diff = trans_obs_y - nn_y;
      double b = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);
      mvGd *= a * exp(-b);
      
    }
    
    // Update particle weights with combined multi-variate Gaussian distribution
    particles[i].weight = mvGd;
    weights[i] = particles[i].weight;

  }
  

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  /*
  std::discrete_distribution produces random integers on the interval [0, n), where the probability of each individual integer i is defined as 
  wi/S, that is the weight of the ith integer divided by the sum of all n weights.
  */

    // Vector for new particles
    vector<Particle> new_particles (num_particles);
    discrete_distribution<int> index(weights.begin(), weights.end());
  
    // Use discrete distribution to return particles by weight
    //random_device rd;
    //default_random_engine gen(rd());
    for (int i = 0; i < num_particles; ++i) {
      
      new_particles[i] = particles[index(gen)];
    
    }
  
    // Replace old particles with the resampled particles
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
