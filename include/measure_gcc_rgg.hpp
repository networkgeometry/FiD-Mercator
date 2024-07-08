#ifndef MEASURE_GCC_RGG
#define MEASURE_GCC_RGG

#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include "incbeta.h"

const static double PI = 3.141592653589793238462643383279502884197;

class MeasureGCC
{
private:
  int nb_vertices;
  int dimension;
  int ntimes_calculate_occupied_surface;
  std::vector<std::vector<double>> positions;
  std::vector<double> thetas;
  std::vector<std::set<int>> adjacency_list;
  double step_theta;
  bool isRadius{false};
  double init_theta{0.001};
  double end_theta{PI/2};

  int get_root(int i, std::vector<int> &clust_id);
  void merge_clusters(std::vector<int> &size, std::vector<int> &clust_id);
  std::pair<double, std::vector<double>>  get_connected_component_size();
  double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2);
  void connect_random_geometric_graph(double theta);
  void connect_random_geometric_graph_radius(double radius);
  bool checkRadius(std::vector<double> v1, std::vector<double> v2, double radius);
  double measure_occupied_surface(const std::vector<double> &vertex2Prop, double theta, int &n_circles, double &scaled_surface_per_circle);
  double measure_occupied_surface_avg(const std::vector<double> &vertex2Prop, double theta, int ntimes, double &avg_n_circles, double &avg_scaled_surface_per_circle);
  double get_surface_per_circle(double theta);

public:
  MeasureGCC() = default;
  MeasureGCC(int dim, int ntimes, const std::string &positions_filename);
  void gcc_vs_theta_relation();
  void setThetaStep(double step_theta) { this->step_theta = step_theta; }
  void setIsRadius(bool isRadius) { this->isRadius = isRadius; }
  void setNtimes(int ntimes) { this->ntimes_calculate_occupied_surface = ntimes; }
  void setInitTheta(double theta) { this->init_theta = theta; }
  void setEndTheta(double theta) { this->end_theta = theta; }
};

MeasureGCC::MeasureGCC(int dim, int ntimes, const std::string &positions_filename) : dimension(dim), ntimes_calculate_occupied_surface(ntimes)
{
  std::stringstream one_line;
  std::string full_line, name1_str;
  nb_vertices = 0;
  std::fstream positions_file(positions_filename.c_str(), std::fstream::in);
  if (!positions_file.is_open())
  {
    std::cerr << "Could not open file: " << positions_filename << "." << std::endl;
    std::terminate();
  }
  while (!positions_file.eof())
  {
    std::getline(positions_file, full_line);
    positions_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
 
    if (dimension == 1)
    {
      one_line >> name1_str >> std::ws;
      thetas.push_back(std::stod(name1_str));
    }
    else
    {
      std::vector<double> tmp_position;
      for (int i = 0; i < dimension + 1; ++i)
      {
        one_line >> name1_str >> std::ws;
        tmp_position.push_back(std::stod(name1_str));
      }
      positions.push_back(tmp_position);
    }
    ++nb_vertices;
    one_line.clear();
  }
  positions_file.close();
}

void MeasureGCC::connect_random_geometric_graph(double theta)
{
  adjacency_list.clear();
  adjacency_list.resize(nb_vertices);
  for (int v1 = 0; v1 < nb_vertices; ++v1)
  {
    for (int v2 = v1 + 1; v2 < nb_vertices; ++v2)
    {
      double angle = 0.0;
      if (dimension == 1)
      {
        angle = PI - std::fabs(PI - std::fabs(thetas[v1] - thetas[v2]));
      }
      else
      {
        angle = compute_angle_d_vectors(positions[v1], positions[v2]);
      }
      if (angle < theta)
      {
        adjacency_list[v1].insert(v2);
        adjacency_list[v2].insert(v1);
      }
    }
  }
}

void MeasureGCC::connect_random_geometric_graph_radius(double radius)
{
  adjacency_list.clear();
  adjacency_list.resize(nb_vertices);
  for (int v1 = 0; v1 < nb_vertices; ++v1)
  {
    for (int v2 = v1 + 1; v2 < nb_vertices; ++v2)
    {  
      std::vector<double> pos1;
      std::vector<double> pos2;
      if (dimension == 1)
      {
        pos1 = {thetas[v1]};
        pos2 = {thetas[v2]};
      } else {
        pos1 = positions[v1];
        pos2 = positions[v2];
      }
      if (checkRadius(pos1, pos2, radius)) {
        adjacency_list[v1].insert(v2);
        adjacency_list[v2].insert(v1);
      }
    }
  }
}

bool MeasureGCC::checkRadius(std::vector<double> v1, std::vector<double> v2, double radius)
{
  double res = 0;
  for (int i = 0; i < v1.size(); ++i)
    res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  return std::sqrt(res) < radius;
}

int MeasureGCC::get_root(int i, std::vector<int> &clust_id)
{
  while (i != clust_id[i])
  {
    clust_id[i] = clust_id[clust_id[i]];
    i = clust_id[i];
  }
  return i;
}

void MeasureGCC::merge_clusters(std::vector<int> &size, std::vector<int> &clust_id)
{
  int v1, v2, v3, v4;
  std::set<int>::iterator it, end;
  // Loops over the vertices.
  for (int i(0); i < nb_vertices; ++i)
  {
    // Loops over the neighbors.
    it = adjacency_list[i].begin();
    end = adjacency_list[i].end();
    for (; it != end; ++it)
    {
      if (get_root(i, clust_id) != get_root(*it, clust_id))
      {
        // Adjust the root of vertices.
        v1 = i;
        v2 = *it;
        if (size[v2] > size[v1])
          std::swap(v1, v2);
        v3 = get_root(v1, clust_id);
        v4 = get_root(v2, clust_id);
        clust_id[v4] = v3;
        size[v3] += size[v4];
      }
    }
  }
}

std::pair<double, std::vector<double>> MeasureGCC::get_connected_component_size()
{
  // Vector containing the ID of the component to which each node belongs.
  std::vector<double> Vertex2Prop(nb_vertices, -1);

  // Vector containing the size of the components.
  std::vector<int> connected_components_size;

  // Set ordering the component according to their size.
  std::set<std::pair<int, int>> ordered_connected_components;

  // Starts with every vertex as an isolated cluster.
  std::vector<int> clust_id(nb_vertices);
  std::vector<int> clust_size(nb_vertices, 1);
  for (int v(0); v < nb_vertices; ++v)
  {
    clust_id[v] = v;
  }
  // Merges clusters until the minimal set is obtained.
  merge_clusters(clust_size, clust_id);
  clust_size.clear();
  // Identifies the connected component to which each vertex belongs.
  int nb_conn_comp = 0;
  int comp_id;
  std::map<int, int> CompID;
  for (int v(0); v < nb_vertices; ++v)
  {
    comp_id = get_root(v, clust_id);
    if (CompID.find(comp_id) == CompID.end())
    {
      CompID[comp_id] = nb_conn_comp;
      connected_components_size.push_back(0);
      ++nb_conn_comp;
    }
    Vertex2Prop[v] = CompID[comp_id];
    connected_components_size[CompID[comp_id]] += 1;
  }

  // Orders the size of the components.
  for (int c(0); c < nb_conn_comp; ++c)
  {
    ordered_connected_components.insert(std::make_pair(connected_components_size[c], c));
  }

  int lcc_id = (--ordered_connected_components.end())->second;
  double lcc_size = (--ordered_connected_components.end())->first;
  return std::pair(lcc_size / nb_vertices, Vertex2Prop);
}

double MeasureGCC::compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2)
{
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i)
  {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);

  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < 1e-10)
    return 1; // the same vectors
  else
    return std::acos(result);
}

void MeasureGCC::gcc_vs_theta_relation()
{
  for (double theta = init_theta; theta < end_theta; theta += step_theta)
  {
    if (isRadius)
    {
      double r = theta;
      connect_random_geometric_graph_radius(r);
    }
    else
    {
      connect_random_geometric_graph(theta);
    }
    auto [gcc, vertex2Prop] = get_connected_component_size();
    double avg_n_circles, avg_scaled_surface_per_circle;
    auto scaled_surface = measure_occupied_surface_avg(vertex2Prop, theta, ntimes_calculate_occupied_surface, avg_n_circles, avg_scaled_surface_per_circle);
    std::cout << theta << " " << gcc << " " << scaled_surface << " " << avg_n_circles << " " << avg_scaled_surface_per_circle << std::endl;
  }
}


double MeasureGCC::measure_occupied_surface(const std::vector<double> &vertex2Prop, double theta, int &n_circles, double &scaled_surface_per_circle) {

  // Random initial node (seed)
  std::vector<int> nodes_order;
  nodes_order.resize(nb_vertices);
  for (int i=0; i<nb_vertices; ++i)
    nodes_order[i] = i;

  auto rd = std::random_device {}; 
  auto rng = std::default_random_engine { rd() };
  std::shuffle(std::begin(nodes_order), std::end(nodes_order), rng);
  
  std::vector<int> visited;
  n_circles = 0;

  for (auto n :nodes_order) {
    if (std::find(visited.begin(), visited.end(), n) == visited.end()) { // only consider not visited nodes
      if (vertex2Prop[n] == 0) { // if this node is in GCC
        // Loops over the neighbors.
        auto it = adjacency_list[n].begin();
        auto end = adjacency_list[n].end();
        for (; it != end; ++it)
          visited.push_back(*it);
        
        visited.push_back(n);
        n_circles++;
      }
    }
  }

  double surface_per_circle = get_surface_per_circle(theta);
  scaled_surface_per_circle = surface_per_circle / nb_vertices;
  return (double) n_circles * scaled_surface_per_circle;
}

double MeasureGCC::measure_occupied_surface_avg(const std::vector<double> &vertex2Prop, double theta, int ntimes, double &avg_n_circles, double &avg_scaled_surface_per_circle) {
  double scaled_surface = 0;
  int n_cicles = 0;
  double scaled_surface_per_circle = 0;
  avg_n_circles = 0;
  avg_scaled_surface_per_circle = 0;
  for (int i=0; i<ntimes; ++i) {
    scaled_surface += measure_occupied_surface(vertex2Prop, theta, n_cicles, scaled_surface_per_circle);
    avg_n_circles += n_cicles;
    avg_scaled_surface_per_circle += scaled_surface_per_circle;
  }
  avg_n_circles /= ntimes;
  avg_scaled_surface_per_circle /= ntimes;
  return scaled_surface / ntimes;
}

double MeasureGCC::get_surface_per_circle(double theta) {
  const double x = std::sin(theta) * std::sin(theta);
  const double a = dimension/2.0;
  const double b = 0.5;
  return 0.5 * nb_vertices * incbeta(a, b, x);
}

#endif // MEASURE_GCC_RGG