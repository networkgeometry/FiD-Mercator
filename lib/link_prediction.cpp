#include <iostream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <complex>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <list>

using VectorOfPairs = std::vector<std::pair<int, int>>;
using SetOfPairs = std::set<std::pair<int, int>>;

std::vector<std::string> Num2Name;
std::map<std::string, int> Name2Num;
VectorOfPairs edgelist;


void print_help() {
    constexpr auto help_message = R"(
        ./link_prediction [dim] [beta] [mu] [path_to_coords] [path_to_edgelist] [roc_curve_resolution] [test_size]
        
        dim                          -- dimension of S^D model
        beta                         -- value of beta which controls clustering coefficient
        mu                           -- value of mu which controls mean degree
        path_to_coords               -- path to the file with the nodes' coordinates (.inf_coord)
        path_to_edgelist             -- path to the edgelist (.edge) 
        roc_curve_resolution         -- number of points in ROC curve (number of lambda values) (default: 50)
        test_size                    -- size of the test size (default: 0.5)
    )";
    std::cout << help_message << std::endl;
}


void load_coords(int dim, 
                 const std::string &coords_path, 
                 std::vector<double> &kappas, 
                 std::vector<double> &thetas, 
                 std::vector<std::vector<double>> &positions) {

    std::stringstream one_line;
    std::string full_line, name1_str, name2_str, name3_str;

    std::fstream hidden_variables_file(coords_path.c_str(), std::fstream::in);
    if( !hidden_variables_file.is_open() )
    {
        std::cerr << "Could not open file: " << coords_path << "." << std::endl;
        std::terminate();
    }

    int n_nodes = 0;
    while(!hidden_variables_file.eof()) {
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

        Num2Name.push_back(name1_str);
        Name2Num[name1_str] = n_nodes;
        n_nodes++;

        if (dim == 1) {
            one_line >> name3_str >> std::ws;
            kappas.push_back(std::stod(name3_str));
            one_line >> name3_str >> std::ws;
            thetas.push_back(std::stod(name3_str));
        } else {
            one_line >> name3_str >> std::ws;
            kappas.push_back(std::stod(name3_str));

            std::vector<double> tmp_position;
            for (int i=0; i<dim+1; ++i) {
                one_line >> name3_str >> std::ws;
                tmp_position.push_back(std::stod(name3_str));
            }
            positions.push_back(tmp_position);
        }
        one_line.clear();
    }
    hidden_variables_file.close();
}


void load_edgelist(const std::string& edgelist_path) {
    std::stringstream one_line;
    std::string full_line, source_str, target_str;

    std::fstream edgelist_file(edgelist_path.c_str(), std::fstream::in);
    if( !edgelist_file.is_open() )
    {
        std::cerr << "Could not open file: " << edgelist_path << "." << std::endl;
        std::terminate();
    }

    while(!edgelist_file.eof()) {
        std::getline(edgelist_file, full_line);
        edgelist_file >> std::ws;
        std::istringstream ss(full_line);
        ss >> source_str >> target_str;

        if(source_str == "#" || target_str == "#")
            continue; 
    
        // Assumption: graph is undirected
        edgelist.push_back(std::make_pair(Name2Num[source_str], Name2Num[target_str]));   
    }
}

bool is_in_vector(VectorOfPairs vec, std::pair<int, int> value) {
    return (std::find(vec.begin(), vec.end(), value) != vec.end());
}

std::vector<std::pair<int, int>> get_all_nodes_pairs(int size) {
    std::vector<std::pair<int, int>> all_pairs;
    for (int i=0; i<size; ++i) {
        for (int j=0; j<i; ++j) {
            auto p = std::make_pair(i, j);
            all_pairs.push_back(p);
        }
    }
    return all_pairs;
}

std::pair<VectorOfPairs, VectorOfPairs> train_test_split(double test_size) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::vector<int> index_vector;
    VectorOfPairs edge_train, edge_test;
    for (int i=0; i<edgelist.size(); ++i)
        index_vector.push_back(i);
    std::shuffle(std::begin(index_vector), std::end(index_vector), rng);

    int test_subset_size = int(test_size * edgelist.size());
    for (int i=0; i<test_subset_size; ++i)
        edge_test.push_back(edgelist[index_vector[i]]);

    for (int i=test_subset_size; i<edgelist.size(); ++i)
        edge_train.push_back(edgelist[index_vector[i]]);

    return std::make_pair(edge_train, edge_test);
}

VectorOfPairs reverse_pairs(const VectorOfPairs& vec) {
    VectorOfPairs reversed_vec;
    for (auto p: vec)
        reversed_vec.push_back(std::make_pair(p.second, p.first));
    return reversed_vec;
}

std::pair<VectorOfPairs, VectorOfPairs> generate_all_missing_links(const VectorOfPairs &all_nodes_pairs, const VectorOfPairs &edges_train, VectorOfPairs edges_test) {
    SetOfPairs all_nodes_pairs_set(all_nodes_pairs.begin(), all_nodes_pairs.end());

    SetOfPairs edges_train_set(edges_train.begin(), edges_train.end());
    auto reversed_edges_train = reverse_pairs(edges_train);
    SetOfPairs reversed_edges_train_set(reversed_edges_train.begin(), reversed_edges_train.end());
    edges_train_set.merge(reversed_edges_train_set);

    SetOfPairs edges_test_set(edges_test.begin(), edges_test.end());
    auto reversed_edges_test = reverse_pairs(edges_test);
    SetOfPairs reversed_edges_test_set(reversed_edges_test.begin(), reversed_edges_test.end());
    edges_test_set.merge(reversed_edges_test_set);


    SetOfPairs unconnected_nodes_pairs_set;
    std::set_difference(all_nodes_pairs_set.begin(), all_nodes_pairs_set.end(), 
                        edges_train_set.begin(), edges_train_set.end(),
                        std::inserter(unconnected_nodes_pairs_set, unconnected_nodes_pairs_set.end()));
    VectorOfPairs unconnected_nodes_pairs(unconnected_nodes_pairs_set.begin(), 
                                          unconnected_nodes_pairs_set.end());

    // VectorOfPairs unconnected_nodes_pairs;
    // for (const auto pair: all_nodes_pairs) {
    //     if (is_in_vector(edges_train, pair))
    //         continue;
    //     auto reversed_pair = std::make_pair(pair.second, pair.first);
    //     if (is_in_vector(edges_train, reversed_pair))
    //         continue;
    //     unconnected_nodes_pairs.push_back(pair);
    // }

    SetOfPairs non_links_set, result2;
    std::set_difference(unconnected_nodes_pairs_set.begin(), unconnected_nodes_pairs_set.end(), 
                        edges_test_set.begin(), edges_test_set.end(),
                        std::inserter(non_links_set, non_links_set.end()));
    VectorOfPairs non_links(non_links_set.begin(), non_links_set.end());
    
    // VectorOfPairs non_links;
    // for (const auto pair: unconnected_nodes_pairs) {
    //     if (is_in_vector(edges_test, pair))
    //         continue;
    //     auto reversed_pair = std::make_pair(pair.second, pair.first);
    //     if (is_in_vector(edges_test, reversed_pair))
    //         continue;
    //     non_links.push_back(pair);
    // }
    
    return std::make_pair(unconnected_nodes_pairs, non_links);
}

std::vector<double> get_connection_probability_S1(const VectorOfPairs &edges, 
                                                  const std::vector<double> &kappas,
                                                  const std::vector<double> &thetas,
                                                  double beta,
                                                  double mu) {
    std::vector<double> probs;                                            
    const int N = Name2Num.size();
    for (auto [source, target]: edges) {
        const auto angle = M_PI - std::fabs(M_PI - std::fabs(thetas[source] - thetas[target]));
        const auto d = N / (2 * M_PI) * angle;
        const double p = 1 / (1 + std::pow(d / (mu * kappas[source] * kappas[target]), beta));
        probs.push_back(p);
    }
    return probs;
}

double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i) {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  
  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < 1e-15)
    return 0;
  else
    return std::acos(result);
}

inline double compute_radius(int dim, int N)
{
  const auto inside = N / (2 * std::pow(M_PI, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}

std::vector<double> get_connection_probability_SD(int dim,
                                                  const VectorOfPairs &edges, 
                                                  const std::vector<double> &kappas,
                                                  const std::vector<std::vector<double>> &positions,
                                                  double beta,
                                                  double mu) {
    
    const double radius = compute_radius(dim, Name2Num.size());
    std::vector<double> probs;                                          
    for (auto [source, target]: edges) {
        const auto angle = compute_angle_d_vectors(positions[source], positions[target]);
        const auto inside = (radius * angle) / std::pow(mu * kappas[source] * kappas[target], 1.0 / dim);
        const auto p = 1 / (1 + std::pow(inside, beta));
        probs.push_back(p);
    }
    return probs;
}

std::vector<double> get_connection_probability(int dim,
                                               const VectorOfPairs &edges, 
                                               const std::vector<double> &kappas, 
                                               const std::vector<double> &thetas, 
                                               const std::vector<std::vector<double>> &positions,
                                               double beta, 
                                               double mu) {
    if (dim == 1) {
        return get_connection_probability_S1(edges, kappas, thetas, beta, mu);
    } else {
        return get_connection_probability_SD(dim, edges, kappas, positions, beta, mu);
    }
}

VectorOfPairs sort_and_take_n(const VectorOfPairs &vec, const std::vector<double> &probs, int n) {
    std::vector<std::pair<std::pair<int, int>, double>> combined_vec;
    for (int i=0; i<vec.size(); ++i)
        combined_vec.push_back(std::make_pair(vec[i], probs[i]));
    
    std::sort(combined_vec.begin(), combined_vec.end(), [](const auto &p1, const auto &p2) {
        return p1.second > p2.second;
    });
    combined_vec.resize(n);
    
    VectorOfPairs output_vec;
    for (auto p: combined_vec)
        output_vec.push_back(p.first);
    return output_vec;
}

int length_set_intersection(VectorOfPairs vec1, VectorOfPairs vec2) {
    SetOfPairs set1(vec1.begin(), vec1.end());
    auto rev_vec1 = reverse_pairs(vec1);
    SetOfPairs rev_set1(rev_vec1.begin(), rev_vec1.end());
    set1.merge(rev_set1);

    SetOfPairs set2(vec2.begin(), vec2.end());
    auto rev_vec2 = reverse_pairs(vec2);
    SetOfPairs rev_set2(rev_vec2.begin(), rev_vec2.end());
    set2.merge(rev_set2);
    
    SetOfPairs output;
    std::set_intersection(set1.begin(), set1.end(), 
                          set2.begin(), set2.end(),
                          std::inserter(output, output.end()));
    return output.size() / 2;
}

int length_set_difference(VectorOfPairs vec1, VectorOfPairs vec2) {
    SetOfPairs set1(vec1.begin(), vec1.end());
    auto rev_vec1 = reverse_pairs(vec1);
    SetOfPairs rev_set1(rev_vec1.begin(), rev_vec1.end());
    set1.merge(rev_set1);

    SetOfPairs set2(vec2.begin(), vec2.end());
    auto rev_vec2 = reverse_pairs(vec2);
    SetOfPairs rev_set2(rev_vec2.begin(), rev_vec2.end());
    set2.merge(rev_set2);
    
    SetOfPairs output;
    std::set_difference(set1.begin(), set1.end(), 
                        set2.begin(), set2.end(),
                        std::inserter(output, output.end()));
    return output.size() / 2;
}

void roc_curve(int dim, 
               double beta, 
               double mu, 
               const std::vector<double> &kappas, 
               const std::vector<double> &thetas, 
               const std::vector<std::vector<double>> &positions, 
               int roc_curve_resolution, 
               double test_size) {
    const int size = Name2Num.size();
    auto all_nodes_pairs = get_all_nodes_pairs(size);

    double step = 1.0 / roc_curve_resolution;
    auto [edges_train, edges_test] = train_test_split(test_size);
    auto omega_R = edges_test;
    auto [omega_bar_E, omega_N] = generate_all_missing_links(all_nodes_pairs, edges_train, edges_test);
    auto prob_omega_bar_E = get_connection_probability(dim, omega_bar_E, kappas, thetas, positions, beta, mu);
        
    std::cout << "lambda,tpr,fpr,pr,rc" << std::endl;
    for (double lambda = 0; lambda <= 1; lambda += step) {
        int lambda_fraction_size = omega_bar_E.size() * lambda;
        auto omega_M = sort_and_take_n(omega_bar_E, prob_omega_bar_E, lambda_fraction_size);

        int TP = length_set_intersection(omega_R, omega_M);
        int FN = length_set_difference(omega_R, omega_M);
        int FP = length_set_intersection(omega_N, omega_M);
        int TN = length_set_difference(omega_N, omega_M);

        double tpr = (double)TP / omega_R.size();
        double fpr = (double)FP / omega_N.size();
        double pr = (double)TP / omega_M.size();
        double rc = tpr;
        std::cout << lambda << "," << tpr << "," << fpr << "," <<  pr << "," << rc << std::endl;       
    }
}

void percentage_missing_true_link_ranking(int dim, 
                                          double beta, 
                                          double mu, 
                                          const std::vector<double> &kappas, 
                                          const std::vector<double> &thetas, 
                                          const std::vector<std::vector<double>> &positions, 
                                          int resolution, 
                                          double test_size) {

    const int size = Name2Num.size();
    auto all_nodes_pairs = get_all_nodes_pairs(size);
    
    auto [edges_train, edges_test] = train_test_split(test_size);
    auto omega_R = edges_test;
    int omega_R_size = omega_R.size();
    auto [omega_bar_E, omega_N] = generate_all_missing_links(all_nodes_pairs, edges_train, edges_test);
    auto prob_omega_bar_E = get_connection_probability(dim, omega_bar_E, kappas, thetas, positions, beta, mu);
    
    const int factor = 10; // 10 times of the edge test size
    double step = 1.0 / resolution;
    std::cout << "x,x_size,test_size,no_predicted_true_links,fraction_predicted_true_links" << std::endl;
    for (double x = 0; x <= factor; x += step) {    
        int x_size = omega_R_size * x;
        auto omega_M = sort_and_take_n(omega_bar_E, prob_omega_bar_E, x_size);
        double no_links = length_set_intersection(omega_R, omega_M);
        std::cout << x << "," << x_size << "," << omega_R_size << "," << no_links << "," << no_links / omega_R_size << std::endl; 
    }
    // Precision = (no_link / omega_R_size) when x = 1
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Error. Wrong number of parameters." << std::endl;
        print_help();
    }

    int dim = std::stoi(argv[1]);
    double beta = std::stod(argv[2]);
    double mu = std::stod(argv[3]);
    std::string coords_path = argv[4];
    std::string edgelist_path = argv[5];

    std::vector<double> kappas;
    std::vector<double> thetas;
    std::vector<std::vector<double>> positions;

    load_coords(dim, coords_path, kappas, thetas, positions);
    load_edgelist(edgelist_path);

    
    int roc_curve_resolution = 50;
    if (argc > 6)
        roc_curve_resolution = std::stoi(argv[6]);

    double test_size = 0.5;
    if (argc > 7)
        test_size = std::stod(argv[7]);

    //roc_curve(dim, beta, mu, kappas, thetas, positions, roc_curve_resolution, test_size);
    percentage_missing_true_link_ranking(dim, beta, mu, kappas, thetas, positions, roc_curve_resolution, test_size);
}
    