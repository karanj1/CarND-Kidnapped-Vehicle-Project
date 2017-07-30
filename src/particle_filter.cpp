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

      //List of Landmarks
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
          predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });   // here predictions encompasses landmarks in particles range
        }

      }  

      // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
      // http://planning.cs.uiuc.edu/node99.html   equuation 3.33
      // https://discussions.udacity.com/t/coordinate-transform/241288/4
      vector<LandmarkObs> transformed_os;
      for (unsigned int q = 0; q < observations.size(); q++) {
        double t_x = cos(p_theta) * observations[q].x - sin(p_theta) * observations[q].y + p_x;
        double t_y = sin(p_theta) * observations[q].x + cos(p_theta) * observations[q].y + p_y;
        transformed_os.push_back(LandmarkObs{ observations[q].id, t_x, t_y });
      }  

      // perform dataAssociation for the predictions and transformed observations on current particle
      dataAssociation(predictions, transformed_os);

      // reinit weight
      particles[i].weight = 1.0;  

      for (unsigned int j = 0; j < transformed_os.size(); j++) {
        
        // placeholders for observation and associated prediction coordinates
        double o_x, o_y, pr_x, pr_y;
        o_x = transformed_os[j].x;
        o_y = transformed_os[j].y;  

        int associated_prediction = transformed_os[j].id;  

        // get the x,y coordinates of the prediction associated with the current observation
        for (unsigned int k = 0; k < predictions.size(); k++) {
          if (predictions[k].id == associated_prediction) {
            pr_x = predictions[k].x;
            pr_y = predictions[k].y;
          }
        }  

        // calculate weight for this observation with multivariate Gaussian  //Lesson 14 Session 13 (Calculating the Particle's Final Weight)
        double s_x = std_landmark[0];
        double s_y = std_landmark[1];
        double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );  

        // product of this obersvation weight with total observations weight
        particles[i].weight *= obs_w;
        weights[i] = particles[i].weight;
      }
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
  
  	// Use discrete distribution to return particles by weight
  	random_device rd;
  	default_random_engine gen(rd());
  	for (int i = 0; i < num_particles; ++i) {
    	discrete_distribution<int> index(weights.begin(), weights.end());
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
