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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//std::normal-distribution  :  https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution

	num_particles = 101;

	 // Engine for later generation of particles
  	random_device rd;
  	default_random_engine gen(rd());

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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

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

	// for each particle...
    for (int i = 0; i < num_particles; i++) {  

      // get the particle x, y coordinates
      double p_x = particles[i].x;
      double p_y = particles[i].y;
      double p_theta = particles[i].theta;  

      // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
      vector<LandmarkObs> predictions;  

      vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
      vector<double> landmark_obs_dist (landmarks.size());
      // for each map landmark...
      for (unsigned int j = 0; j < landmarks.size(); j++) {  

        // get id and x,y coordinates
        float lm_x = landmarks[j].x_f;
        float lm_y = landmarks[j].y_f;
        int lm_id = landmarks[j].id_i;
        
        
        //Rectangular Region : rather than using the "sqrt" method considering a circular region around the particle, 
        //this considers a rectangular region but is computationally faster)
        //if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) { 

        //Circular region using Eucladian distance "sqrt"
        double landmark_part_dist = sqrt(pow(p_x - lm_x, 2) + pow(p_y - lm_y, 2));
        // only consider landmarks within sensor range of the particle
        if (landmark_part_dist <= sensor_range) {
          // add prediction to vector
          predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
        }

      }  

      // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
      vector<LandmarkObs> transformed_os;
      for (unsigned int j = 0; j < observations.size(); j++) {
        double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
        double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
        transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
      }  

      // perform dataAssociation for the predictions and transformed observations on current particle
      dataAssociation(predictions, transformed_os);

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
