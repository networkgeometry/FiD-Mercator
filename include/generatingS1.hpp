/*
 *
 * This class provides the functions to generate a graph in the S1 space.
 *
 * Compilation requires the c++11 standard to use #include <random>.
 *   Example: g++ -O3 -std=c++11 my_code.cpp -o my_program
 *
 * Author:  Antoine Allard
 * WWW:     antoineallard.info
 * Date:    November 2017
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 */

#ifndef GENERATINGS1_HPP_INCLUDED
#define GENERATINGS1_HPP_INCLUDED

// Standard Template Library
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>



class generatingS1_t
{
  // Flags controlling options.
  public:
    bool CUSTOM_OUTPUT_ROOTNAME_MODE = false;
    bool NAME_PROVIDED = false;
    bool NATIVE_INPUT_FILE = false;
    bool THETA_PROVIDED = false;
    bool OUTPUT_VERTICES_PROPERTIES = false;
  // Global parameters.
  public:
    // Random number generator seed.
    int SEED = std::time(NULL);
    // Parameter beta (clustering).
    double BETA = -1;
    // Parameter mu (average degree).
    double MU = -1;
    // Rootname for the output files;
    std::string OUTPUT_ROOTNAME = "default_output_rootname";
    // Input hidden variables filename.
    std::string HIDDEN_VARIABLES_FILENAME;
    // Dimension of the model S^D
    int DIMENSION = 1;
  // General internal objects.
  private:
    // pi
    const double PI = 3.141592653589793238462643383279502884197;
    const double NUMERICAL_ZERO = 1e-10;
    // Random number generator
    std::mt19937 engine;
    std::uniform_real_distribution<double> uniform_01;
    std::normal_distribution<double> normal_01;
    // Mapping the numerical ID of vertices to their name.
    std::vector<std::string> Num2Name;
  // Objects related to the graph ensemble.
  private:
    // Number of vertices.
    int nb_vertices;
    // Hidden variables of the vertices.
    std::vector<double> kappa;
    // Positions of the vertices.
    std::vector<double> theta;
    // Position of the vertices in D-dimensions
    std::vector<std::vector<double>> d_positions;
  // Public functions to generate the graphs.
  public:
    // Constructor (empty).
    generatingS1_t() {};
    // Loads the values of the hidden variables (i.e., kappa and theta).
    void load_hidden_variables();
    void load_hidden_variables_dim();
    // Generates an edgelist and writes it into a file.
    void generate_edgelist(int width = 15);
    void generate_edgelist_dim(int width = 15); 
  // Private functions linked to the generation of a random edgelist.
  private:
    // Saves the values of the hidden variables (i.e., kappa and theta).
    void save_vertices_properties(std::vector<int>& rdegree, std::vector<double>& edegree, int width);
    void save_vertices_properties_dim(std::vector<int>& rdegree, std::vector<double>& edegree, int width);
    // Generate random coordiantes in D dimensional space
    std::vector<double> generate_random_d_vector(int dim);
    double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2);
    inline double compute_radius(int dim, int N) const;

    void fix_error_with_zero_degree_nodes(std::vector<int>& rdegree, std::vector<double>& edegree);
    // Gets and format current date/time.
    std::string get_time();
};





// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void generatingS1_t::generate_edgelist(int width)
{
  // Initializes the random number generator.
  engine.seed(SEED);
  // Sets the name of the file to write the edgelist into.
  std::string edgelist_filename = OUTPUT_ROOTNAME + ".edge";
  // Vectors containing the expected and real degrees.
  std::vector<double> edegree;
  std::vector<int> rdegree;
  // Initializes the containers for the expected and real degrees.
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    edegree.resize(nb_vertices, 0);
    rdegree.resize(nb_vertices, 0);
  }
  // Makes sure the value of beta has been provided.
  if(BETA < 0)
  {
    std::cerr << "ERROR: The value of parameter beta must be provided." << std::endl;
    std::terminate();
  }
  fix_error_with_zero_degree_nodes(rdegree, edegree);
  // Sets the value of mu, if not provided.
  if(MU < 0)
  {
    // Computes the average value of kappa.
    double average_kappa = 0;
    for(int v(0); v<nb_vertices; ++v)
    {
      average_kappa += kappa[v];
    }
    average_kappa /= nb_vertices;
    // Sets the "default" value of mu.
    MU = BETA * std::sin(PI / BETA) / (2.0 * PI * average_kappa);
  }
  // Generates the values of theta, if not provided.
  if(theta.size() != nb_vertices)
  {
    theta.clear();
    theta.resize(nb_vertices);
    for(int v(0); v<nb_vertices; ++v)
    {
      theta[v] = 2 * PI * uniform_01(engine);
    }
  }
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream edgelist_file(edgelist_filename.c_str(), std::fstream::out);
  if( !edgelist_file.is_open() )
  {
    std::cerr << "ERROR: Could not open file: " << edgelist_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "# Generated on:           " << get_time()                << std::endl;
  edgelist_file << "# Hidden variables file:  " << HIDDEN_VARIABLES_FILENAME << std::endl;
  edgelist_file << "# Seed:                   " << SEED                      << std::endl;
  edgelist_file << "#"                                                       << std::endl;
  edgelist_file << "# Parameters"                                            << std::endl;
  edgelist_file << "#   - nb. vertices:       " << nb_vertices               << std::endl;
  edgelist_file << "#   - beta:               " << BETA                      << std::endl;
  edgelist_file << "#   - mu:                 " << MU                        << std::endl;
  edgelist_file << "#   - radius:             " << nb_vertices / (2 * PI)    << std::endl;
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "#";
  edgelist_file << std::setw(width - 1) << "Vertex1" << " ";
  edgelist_file << std::setw(width)     << "Vertex2" << " ";
  edgelist_file << std::endl;
  // Generates the edgelist.
  double kappa1, theta1, dtheta, prob;
  double prefactor = nb_vertices / (2 * PI * MU);
  for(int v1(0); v1<nb_vertices; ++v1)
  {
    kappa1 = kappa[v1];
    theta1 = theta[v1];
    for(int v2(v1 + 1); v2<nb_vertices; ++v2)
    {
      dtheta = PI - std::fabs(PI - std::fabs(theta1 - theta[v2]));
      prob = 1 / (1 + std::pow((prefactor * dtheta) / (kappa1 * kappa[v2]), BETA));
      if(uniform_01(engine) < prob)
      {
        edgelist_file << std::setw(width) << Num2Name[v1] << " ";
        edgelist_file << std::setw(width) << Num2Name[v2] << " ";
        edgelist_file << std::endl;
        if(OUTPUT_VERTICES_PROPERTIES)
        {
          rdegree[v1] += 1;
          rdegree[v2] += 1;
        }
      }
      if(OUTPUT_VERTICES_PROPERTIES)
      {
        edegree[v1] += prob;
        edegree[v2] += prob;
      }
    }
  }
  // Closes the stream.
  edgelist_file.close();
  // Outputs the hidden variables, if required.
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    save_vertices_properties(rdegree, edegree, width);
  }
}

void generatingS1_t::generate_edgelist_dim(int width)
{
  const auto inside = nb_vertices / (2 * std::pow(PI, (DIMENSION + 1) / 2.0)) * std::tgamma((DIMENSION + 1) / 2.0);
  const double radius = std::pow(inside, 1.0 / DIMENSION);

 // Initializes the random number generator.
  engine.seed(SEED);
  // Sets the name of the file to write the edgelist into.
  std::string edgelist_filename = OUTPUT_ROOTNAME + ".edge";
  // Vectors containing the expected and real degrees.
  std::vector<double> edegree;
  std::vector<int> rdegree;
  // Initializes the containers for the expected and real degrees.
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    edegree.resize(nb_vertices, 0);
    rdegree.resize(nb_vertices, 0);
  }
  // Makes sure the value of beta has been provided.
  if(BETA < 0)
  {
    std::cerr << "ERROR: The value of parameter beta must be provided." << std::endl;
    std::terminate();
  }
  fix_error_with_zero_degree_nodes(rdegree, edegree);
  // Sets the value of mu, if not provided.
  if(MU < 0)
  {
    // Computes the average value of kappa.
    double average_kappa = 0;
    for(int v(0); v<nb_vertices; ++v)
    {
      average_kappa += kappa[v];
    }
    average_kappa /= nb_vertices;
    // Sets the "default" value of mu.
    const auto top = BETA * std::tgamma(DIMENSION / 2.0) * std::sin(DIMENSION * PI / BETA);
    const auto bottom = average_kappa * 2 * std::pow(PI, 1 + DIMENSION / 2.0);
    MU =  top / bottom;
  }
  // Generates random positions of nodes only if not specified previously
  if (d_positions.size() < 1) {
    d_positions.clear();
    d_positions.resize(nb_vertices);
    for(int v(0); v<nb_vertices; ++v)
      d_positions[v] = generate_random_d_vector(DIMENSION);
  }
  
  // std::string file = edgelist_filename.substr(0, edgelist_filename.find("."));
  // file.append(".coords");
  // std::ofstream outfile(file);
  // std::for_each(d_positions.begin(), d_positions.end(), [&](auto x){
  //   for (int i=0; i<DIMENSION+1; ++i)
  //     outfile << x[i] << '\t';
  //   outfile << '\n';
  // });
  // outfile.close();

  // Opens the stream and terminates if the operation did not succeed.
  std::fstream edgelist_file(edgelist_filename.c_str(), std::fstream::out);
  if( !edgelist_file.is_open() )
  {
    std::cerr << "ERROR: Could not open file: " << edgelist_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "# Generated on:           " << get_time()                << std::endl;
  edgelist_file << "# Hidden variables file:  " << HIDDEN_VARIABLES_FILENAME << std::endl;
  edgelist_file << "# Seed:                   " << SEED                      << std::endl;
  edgelist_file << "#"                                                       << std::endl;
  edgelist_file << "# Parameters"                                            << std::endl;
  edgelist_file << "#   - nb. vertices:       " << nb_vertices               << std::endl;
  edgelist_file << "#   - beta:               " << BETA                      << std::endl;
  edgelist_file << "#   - mu:                 " << MU                        << std::endl;
  edgelist_file << "#   - radius:             " << radius                    << std::endl;
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "#";
  edgelist_file << std::setw(width - 1) << "Vertex1" << " ";
  edgelist_file << std::setw(width)     << "Vertex2" << " ";
  edgelist_file << std::endl;
  // Generates the edgelist.
  for(int v1(0); v1<nb_vertices; ++v1) {
    for(int v2(v1 + 1); v2<nb_vertices; ++v2) {
      const auto dtheta = compute_angle_d_vectors(d_positions[v1], d_positions[v2]);
      const auto inside = radius * dtheta / std::pow(MU * kappa[v1] * kappa[v2], 1.0 / DIMENSION);
      const auto prob = 1 / (1 + std::pow(inside, BETA));
      if(uniform_01(engine) < prob)
      {
        edgelist_file << std::setw(width) << Num2Name[v1] << " ";
        edgelist_file << std::setw(width) << Num2Name[v2] << " ";
        edgelist_file << std::endl;
        if(OUTPUT_VERTICES_PROPERTIES)
        {
          rdegree[v1] += 1;
          rdegree[v2] += 1;
        }
      }
      if(OUTPUT_VERTICES_PROPERTIES)
      {
        edegree[v1] += prob;
        edegree[v2] += prob;
      }
    }
  }
  // Closes the stream.
  edgelist_file.close();
  // Outputs the hidden variables, if required.
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    save_vertices_properties_dim(rdegree, edegree, width);
  }
}



// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void generatingS1_t::load_hidden_variables()
{
  // Stream object.
  std::stringstream one_line;
  // String objects.
  std::string full_line, name1_str, name2_str, name3_str;
  // Resets the number of vertices.
  nb_vertices = 0;
  // Resets the container.
  kappa.clear();
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream hidden_variables_file(HIDDEN_VARIABLES_FILENAME.c_str(), std::fstream::in);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << HIDDEN_VARIABLES_FILENAME << "." << std::endl;
    std::terminate();
  }
  // Extracts the beta and mu parameters if the file is a native.
  if(NATIVE_INPUT_FILE)
  {
    // Ignores the first 9 lines of the file.
    for(int l(0); l<8; ++l)
    {
      std::getline(hidden_variables_file, full_line);
    }
    // Gets the 10th lines containing the value of beta.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    BETA = std::stod(name1_str);
    one_line.clear();
    // Gets the 11th lines containing the value of mu.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    MU = std::stod(name1_str);
    one_line.clear();
  }
  // Reads the hidden variables file line by line.
  while( !hidden_variables_file.eof() )
  {
    // Reads a line of the file.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    // Skips lines of comment.
    if(name1_str == "#")
    {
      one_line.clear();
      continue;
    }
    // Adds the new vertex and its hidden variable(s).
    if(NAME_PROVIDED)
    {
      one_line >> name2_str >> std::ws;
      Num2Name.push_back(name1_str);
      kappa.push_back(std::stod(name2_str));
    }
    else
    {
      Num2Name.push_back("v" + std::to_string(nb_vertices));
      kappa.push_back(std::stod(name1_str));
    }
    if(THETA_PROVIDED)
    {
      one_line >> name3_str >> std::ws;
      theta.push_back(std::stod(name3_str));
    }
    ++nb_vertices;
    one_line.clear();
  }
  // Closes the stream.
  hidden_variables_file.close();
}


void generatingS1_t::load_hidden_variables_dim()
{
  // Stream object.
  std::stringstream one_line;
  // String objects.
  std::string full_line, name1_str, name2_str, name3_str;
  // Resets the number of vertices.
  nb_vertices = 0;
  // Resets the container.
  kappa.clear();
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream hidden_variables_file(HIDDEN_VARIABLES_FILENAME.c_str(), std::fstream::in);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << HIDDEN_VARIABLES_FILENAME << "." << std::endl;
    std::terminate();
  }
  if(NATIVE_INPUT_FILE)
  {
    // Ignores the first 9 lines of the file.
    for(int l(0); l<8; ++l)
    {
      std::getline(hidden_variables_file, full_line);
    }
    // Gets the 10th lines containing the value of beta.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    BETA = std::stod(name1_str);
    one_line.clear();
    // Gets the 11th lines containing the value of mu.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    one_line >> name1_str >> std::ws;
    MU = std::stod(name1_str);
    one_line.clear();
  }
  // Reads the hidden variables file line by line.
  while( !hidden_variables_file.eof() )
  {
    // Reads a line of the file.
    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;
    // Skips lines of comment.
    if(name1_str == "#")
    {
      one_line.clear();
      continue;
    }
    // Adds the new vertex and its hidden variable(s).
    if(NAME_PROVIDED)
    {
      one_line >> name2_str >> std::ws;
      Num2Name.push_back(name1_str);
      kappa.push_back(std::stod(name2_str));
    }
    else
    {
      Num2Name.push_back("v" + std::to_string(nb_vertices));
      kappa.push_back(std::stod(name1_str));
    }

    if (THETA_PROVIDED) {
      /*
      If file looks like this:

      kappa pos.1 pos.2 ... (up to pos.D+1)
      */
      std::vector<double> tmp_position;
      for (int i=0; i<DIMENSION+1; ++i) {
        one_line >> name3_str >> std::ws;
        tmp_position.push_back(std::stod(name3_str));
      }
      d_positions.push_back(tmp_position);
    }
  
    ++nb_vertices;
    one_line.clear();
  }
  // Closes the stream.
  hidden_variables_file.close();
}

void generatingS1_t::save_vertices_properties_dim(std::vector<int>& rdegree, std::vector<double>& edegree, int width)
{
  // Finds the minimal value of kappa.
  double kappa_min = *std::min_element(kappa.begin(), kappa.end());
  // Sets the name of the file to write the hidden variables into.
  std::string hidden_variables_filename = OUTPUT_ROOTNAME + ".gen_coord";
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream hidden_variables_file(hidden_variables_filename.c_str(), std::fstream::out);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << hidden_variables_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  hidden_variables_file << "#";
  hidden_variables_file << std::setw(width - 1) << "Vertex"   << " ";
  hidden_variables_file << std::setw(width)     << "Kappa"    << " ";
  for (int i=0; i<DIMENSION+1; ++i)
    hidden_variables_file << std::setw(width)   << "Pos." << i << " ";  
  hidden_variables_file << std::setw(width)     << "RealDeg." << " ";
  hidden_variables_file << std::setw(width)     << "Exp.Deg." << " ";
  hidden_variables_file << std::endl;
  // Writes the hidden variables.
  for(int v(0); v<nb_vertices; ++v)
  {
    hidden_variables_file << std::setw(width) << Num2Name[v]                                                    << " ";
    hidden_variables_file << std::setw(width) << kappa[v]                                                       << " ";
    for (int i=0; i<DIMENSION+1; ++i)
      hidden_variables_file << std::setw(width)   << d_positions[v][i] << " ";
    hidden_variables_file << std::setw(width) << rdegree[v]                                                     << " ";
    hidden_variables_file << std::setw(width) << edegree[v]                                                     << " ";
    hidden_variables_file << std::endl;
  }
  // Closes the stream.
  hidden_variables_file.close();
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void generatingS1_t::save_vertices_properties(std::vector<int>& rdegree, std::vector<double>& edegree, int width)
{
  // Finds the minimal value of kappa.
  double kappa_min = *std::min_element(kappa.begin(), kappa.end());
  // Sets the name of the file to write the hidden variables into.
  std::string hidden_variables_filename = OUTPUT_ROOTNAME + ".gen_coord";
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream hidden_variables_file(hidden_variables_filename.c_str(), std::fstream::out);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << hidden_variables_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  hidden_variables_file << "#";
  hidden_variables_file << std::setw(width - 1) << "Vertex"   << " ";
  hidden_variables_file << std::setw(width)     << "Kappa"    << " ";
  hidden_variables_file << std::setw(width)     << "Theta"    << " ";
  hidden_variables_file << std::setw(width)     << "Hyp.Rad." << " ";
  hidden_variables_file << std::setw(width)     << "RealDeg." << " ";
  hidden_variables_file << std::setw(width)     << "Exp.Deg." << " ";
  hidden_variables_file << std::endl;
  // Writes the hidden variables.
  for(int v(0); v<nb_vertices; ++v)
  {
    hidden_variables_file << std::setw(width) << Num2Name[v]                                                    << " ";
    hidden_variables_file << std::setw(width) << kappa[v]                                                       << " ";
    hidden_variables_file << std::setw(width) << theta[v]                                                       << " ";
    hidden_variables_file << std::setw(width) << 2 * std::log( nb_vertices / (PI * MU * kappa_min * kappa[v]) ) << " ";
    hidden_variables_file << std::setw(width) << rdegree[v]                                                     << " ";
    hidden_variables_file << std::setw(width) << edegree[v]                                                     << " ";
    hidden_variables_file << std::endl;
  }
  // Closes the stream.
  hidden_variables_file.close();
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
std::string generatingS1_t::get_time()
{
  // Gets the current date/time.
  time_t theTime = time(NULL);
  struct tm *aTime = gmtime(&theTime);
  int year    = aTime->tm_year + 1900;
  int month   = aTime->tm_mon + 1;
  int day     = aTime->tm_mday;
  int hours   = aTime->tm_hour;
  int minutes = aTime->tm_min;
  // Format the string.
  std::string the_time = std::to_string(year) + "/";
  if(month < 10)
    the_time += "0";
  the_time += std::to_string(month) + "/";
  if(day < 10)
    the_time += "0";
  the_time += std::to_string(day) + " " + std::to_string(hours) + ":";
  if(minutes < 10)
    the_time += "0";
  the_time += std::to_string(minutes) + " UTC";
  // Returns the date/time.
  return the_time;
}


std::vector<double> generatingS1_t::generate_random_d_vector(int dim) {
  std::vector<double> positions;
  positions.resize(dim + 1);
  double norm{0};
  for (auto &pos : positions) {
    pos = normal_01(engine);
    norm += pos * pos;
  }
  norm /= std::sqrt(norm);
  // Normalize vector
  for (auto &pos: positions)
    pos /= norm;

  // Rescale by the radius in a given dimension
  const auto R = compute_radius(dim, nb_vertices);
  for (auto &pos: positions)
    pos *= R;
  return positions;
}

double generatingS1_t::compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i) {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  
  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < NUMERICAL_ZERO)
    return 0; // the same vectors
  else
    return std::acos(result);
}

inline double generatingS1_t::compute_radius(int dim, int N) const
{
  const auto inside = N / (2 * std::pow(PI, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}


void generatingS1_t::fix_error_with_zero_degree_nodes(std::vector<int>& rdegree, std::vector<double>& edegree) {
  // Generated networks are usually smaller than the input ones
  // To solve this issue we propose to add N_0 nodes with kappa values 
  // sampled from the original kappas. In such a way we would obtain the 
  // network with almost the same size and average degree as the input one.
  
  double mean_exp_kappa = 0;
  for (int i=0; i<nb_vertices; ++i) {
    mean_exp_kappa += std::exp(-kappa[i]);
  }
  mean_exp_kappa /= nb_vertices;
  int N_0 = round(nb_vertices * mean_exp_kappa / (1 - mean_exp_kappa));  
  int new_nb_vertices = nb_vertices + N_0;

  std::cout << "Adding N_0 = " << N_0 << " nodes to the original network" << std::endl;  
  std::vector<double> new_kappas;
  std::sample(kappa.begin(), kappa.end(), std::back_inserter(new_kappas), N_0, std::mt19937{std::random_device{}()});

  for (const auto &k: new_kappas)
    kappa.push_back(k);

  for (int i=0; i<N_0; ++i) {
    rdegree.push_back(nb_vertices + i);
    edegree.push_back(nb_vertices + i);
    Num2Name.push_back("v" + std::to_string(nb_vertices + i));
  }
  nb_vertices = new_nb_vertices;
}

#endif // GENERATINGS1_HPP_INCLUDED
