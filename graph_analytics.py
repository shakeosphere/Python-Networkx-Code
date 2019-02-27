
import json
import networkx as nx
import sys
import argparse
import matplotlib.pyplot as plt
from math import log10
import db_shakeosphere as dbs

######################
## HELPER FUNCTIONS ##
######################

# Project a bipartite graph onto one of its left or right sets.
def project_graph(bigraph, project):
    if project == 'left':
        project_nodes = bigraph.left
    elif project == 'right':
        project_nodes = bigraph.right
    else:
        return bigraph
    return nx.bipartite.weighted_projected_graph(bigraph,
                                                 project_nodes)    

# Search through a graph for individuals with a given name. Must be a
# graph of individuals.
def name_to_id(graph, name):
    if not name:
        return None
    
    name_id = None
    for node in graph.nodes():
        if name == graph.node[node]['name']:
            if name_id:
                sys.stderr.write('WARNING: Multiple individuals with ' +
                                 'the name "' + name + '" exist!\n')
            name_id = node
            
    # Make sure that we have found the name in the graph, otherwise throw
    # an error.
    if name and not name_id:
        print('ERROR: Could not find "' + name + '" in the network.')
        exit(1)
    return name_id

#####################################
## PRINTING AND PLOTTING FUNCTIONS ##
#####################################

# Plot scatter plot.
def plot_scatter(x, y, title, xlabel, ylabel, more=None):
    plt.figure()
    plt.grid(True)
    plt.scatter(x, y, c='b')
    if more:
        plt.scatter(x, more, c='r')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Plot line plot.
def plot_line(x, y, title, xlabel, ylabel, more=None):
    plt.figure()
    plt.grid(True)
    plt.plot(x, y, 'bo')
    if more:
        plt.plot(x, more, 'ro')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Plot data distributions.
def plot_distribution(buckets, values, title, xlabel, ylabel):
    plt.figure()
    plt.grid(True)
    plt.bar(buckets, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Plot the data provided as a tab-delimited table, where each successive
# argument corresponds to a new column of data.
def print_table(*cols):
    if len(cols) == 0:
        return
    # Assume that all columns have same number of rows as the first
    # column.
    for idx in range(len(cols[0])):
        for col in cols:
            sys.stdout.write(str(col[idx]))
            sys.stdout.write('\t')
        sys.stdout.write('\n')

# Output a table to the specified JSON file.
def out_json_table(json_file, headers, *cols):
    if len(headers) != len(cols):
        sys.stderr.write('WARNING: The number of column headers ' +
                         'does not match the number of columns.\n')
        return
    # Create JSON object with each header name mapping to a column of
    # data.
    table = {}
    idx = 0
    for col in cols:
        table[headers[idx]] = col
        idx += 1
    json.dump(table, json_file)

# Output x and y coordinates to the specified JSON file.
def out_json_plot(json_file, x, y, title, xlabel, ylabel):
    if len(x) != len(y):
        sys.stderr.write('WARNING: The length of x and y do not match.\n')
        return
    # Create JSON object with fields for x, y, title, xlabel, and ylabel.
    plot = {}
    plot['title'] = title
    plot[xlabel] = x
    plot[ylabel] = y
    json.dump(plot, json_file)

#############################
## GRAPH LOADING FUNCTIONS ##
#############################

# Wrap the NetworkX Graph class to also keep track of the left and right
# sets of nodes in the bipartite graph, which is needed for some of the
# bipartite algorithms.
class BipartiteGraph(nx.Graph):
    def __init__(self):
        super(BipartiteGraph, self).__init__()
        self.left = set([])
        self.right = set([])

# Load NetworkX graph from a database file.
def load_graph_db(years):
    cursor, conn = dbs.connect()
    
    # Initialize NetworkX graph.
    graph = BipartiteGraph()

    roles = [ 'author',
              'printer',
              'publisher',
              'bookseller' ]
    # Load individuals for each role.
    for role in roles:
        for person_id, fname, lname, s in dbs.get_indivs(cursor, role,
                                                         years[0],
                                                         years[-1]):
            if (not fname) or (not lname):
                continue
            name = fname + ' ' + lname
            person_str = str(person_id) + '_indiv'
            graph.add_node(person_str, name=name, role=role, bipartite=0)
            graph.left.add(person_str)

    # Load publications.
    for pub_id, title, year in dbs.get_publications(cursor, years[0],
                                                    years[-1]):
        graph.add_node(pub_id, title=title, bipartite=1, year=year)
        graph.right.add(pub_id)

    # Add edges to the graph.
    for pub_id, person_id, role in dbs.get_person_to_pub(cursor):
        person_str = str(person_id) + '_indiv'        
        if not graph.has_node(person_str):
            continue
        if not graph.has_node(pub_id):
            continue
        graph.add_edge(person_str, pub_id)

        # Separate graph out by year as well.
        year = graph.node[pub_id]['year']
        if not year in load_graph_db.cache:
            load_graph_db.cache[year] = BipartiteGraph()
        year_graph = load_graph_db.cache[year]
        year_graph.add_node(person_str, graph.node[person_str])
        year_graph.add_node(pub_id, graph.node[pub_id])
        year_graph.add_edge(person_str, pub_id)
        year_graph.left.add(person_str)
        year_graph.right.add(pub_id)

    cursor.close()
    conn.close()
    
    return graph
# Cache declared here.
load_graph_db.cache = {}

# Load a bipartite graph from a given range of years, and potentially
# return the projection of that graph onto its left or right set of nodes.
def load_graph(years, project='off'):
    # If all of the years have been loaded, just read from the cache.
    if all([ year in load_graph_db.cache for year in years ]):
        years_graph = BipartiteGraph()
        for year in years:
            curr_graph = load_graph_db.cache[year]
            years_graph.add_nodes_from(curr_graph.nodes(data=True))
            years_graph.add_edges_from(curr_graph.edges(data=True))
            years_graph.left.update(curr_graph.left)
            years_graph.right.update(curr_graph.right)
    else:
        years_graph = load_graph_db(years)
    # If needed, project the graph onto one of its left or right sets.
    return project_graph(years_graph, project)

# Load a bipartite graph from a single year.
def load_graph_year(year, project='off'):
    # Check cache to see if year has already been loaded.
    if year in load_graph_db.cache:
        year_graph = load_graph_db.cache[year]
    else:
        year_graph = load_graph_db([ year ])
    # If needed, project the graph onto one of its left or right sets.
    return project_graph(year_graph, project)

#######################
## RANKING FUNCTIONS ##
#######################
    
# Filter a list of nodes by role.
def filter_role(nodes, role, graph):
    roles = nx.get_node_attributes(graph, 'role')
    filtered = {}
    for node in nodes:
        if roles[node] == role:
            filtered[node] = nodes[node]
    return filtered

# Rank a dictionary's keys according to the numerical values inside.
def rank(dict):
    return sorted(dict, key=dict.get, reverse=True)

# Print a ranking report.
def print_ranking(top_nodes, nodes_to_data, graph, pub_graph=False):
    data = [ nodes_to_data[node] for node in top_nodes ]
    # Graph of publications.
    if pub_graph:
        titles = [ graph.node[node]['title'] for node in top_nodes ]
        years = [ graph.node[node]['year'] for node in top_nodes ]
        print_table(top_nodes, data, titles, years)
    # Graph of individuals.
    else:
        names = [ graph.node[node]['name'] for node in top_nodes ]
        print_table(names, data)

# Print the top number of individuals according to betweenness centrality.
def rank_betweenness(graph, top_n, role='all', pub_graph=False):
    print('\nNormalized betweenness centrality rank:')
    betw_nodes = nx.betweenness_centrality(graph)
    if not pub_graph and role != 'all':
        betw_nodes = filter_role(betw_nodes, role, graph)
    top_nodes = rank(betw_nodes)[0:top_n]
    print_ranking(top_nodes, betw_nodes, graph, pub_graph)
    return betw_nodes

# Rank edges based on edge betweenness centrality.
def rank_edge_betweenness(graph, bigraph, top_n=10,
                          pub_graph=False, plot_on=True):

    #from edgeswap import EdgeSwapGraph
    #graph = EdgeSwapGraph(graph).randomize_by_edge_swaps(500)
    
    print('\nNormalized edge betweenness centrality rank:')
    betw_edges = nx.edge_betweenness_centrality(graph)
    top_edges_betw = rank(betw_edges)

    # Rank the individuals based on betwenness, but store the weights so
    # we can create a plot to (perhaps) demonstrate the strength of weak
    # ties.
    names = []
    arrow = '<--->'
    betweennesses = []
    weights = []
    for n0, n1 in top_edges_betw:
        if pub_graph:
            names0 = graph.node[n0]['title']
            names1 = graph.node[n1]['title']
        else:
            names0 = graph.node[n0]['name']
            names1 = graph.node[n1]['name']
        names.append(arrow.join([ names0, names1 ]))
        betweennesses.append(betw_edges[(n0, n1)])
        weights.append(graph[n0][n1]['weight'])
    print_table(names[0:top_n], betweennesses[0:top_n], weights[0:top_n])
    if plot_on:
        plot_scatter(weights, betweennesses, 'Strength of weak ties',
                     'Edge weight', 'Edge betweenness')

    if len(names) == 0:
        return
    print('Node(s) associated with the top degree edge:')
    pair_common_neighbors(bigraph,
                          top_edges_betw[0][0], top_edges_betw[0][1],
                          names[0].split(arrow)[0],
                          names[0].split(arrow)[1],
                          pub_graph=pub_graph,
                          verbose=False)

# Print the top number of individuals according to degree.
def rank_degree(graph, top_n, role='all', pub_graph=False):
    print('\nDegree rank:')
    degr_nodes = graph.degree()
    if not pub_graph and role != 'all':
        degr_nodes = filter_role(degr_nodes, role, graph)
    top_nodes = rank(degr_nodes)[0:top_n]
    print_ranking(top_nodes, degr_nodes, graph, pub_graph)
    return degr_nodes

# Print the top number of individuals according to the sum of the
# adjacent weights.
def rank_degree_weight(graph, top_n, role='all', pub_graph=False):
    print('\nWeighted degree rank:')
    publ_nodes = graph.degree(weight='weight')
    if not pub_graph and role != 'all':
        publ_nodes = filter_role(publ_nodes, role, graph)
    top_nodes = rank(publ_nodes)[0:top_n]
    print_ranking(top_nodes, publ_nodes, graph, pub_graph)
    return publ_nodes

def rank_link_prediction(source, graph, top_n, role='all', pub_graph=False):
    print('\nLink prediction ranking:')
    if pub_graph:
        source_name = str(source)
    else:
        source_name = graph.node[source]['name']
    print('Source node: ' + source_name)

    neighbors = graph.neighbors(source)
    unconnected_edges = [ (source, node) for node in graph.nodes()
                          if (not node in neighbors) and (source != node) ]
    preds = nx.preferential_attachment(graph, unconnected_edges)
    #preds = nx.adamic_adar_index(graph, unconnected_edges)
    
    pred_edges = {}
    for source, node, score in preds:
        if pub_graph:
            node_name = str(node)
        else:
            node_name = graph.node[node]['name']
        pred_edges[source_name + '<--->' + node_name] = score
        
    top_edges = rank(pred_edges)[0:top_n]
    top_edge_scores = [ pred_edges[e] for e in top_edges  ]
    print_table(top_edges, top_edge_scores)
    return top_edges, top_edge_scores


############################
## DISTRIBUTION FUNCTIONS ##
############################

# Print number of nodes corresponding to each degree value.
def distribution_degree(graph, cutoff, plot_on, json_file=None):
    description = ('Degree distribution of largest connected ' +
                   'component (cutoff = ' + str(cutoff) + ')')
    print('\n' + description + ':')
    print('Degree \tCount')
    largest_cc = max(nx.connected_component_subgraphs(graph),
                     key=len)
    values = nx.degree_histogram(largest_cc)
    values_cut = values[0:cutoff]
    print_table(range(len(values_cut)), values_cut)
    # Plot data.
    if plot_on:
        plot_distribution(range(len(values_cut)), values_cut, description,
                          'Degree', 'Individuals')
        plot_scatter([ log10(d) if d > 0 else float('nan') for d in
                        range(len(values)) ],
                     [ log10(v) if v > 0 else float('nan') for v in values ],
                     'log plot of degree distribution',
                     'log10(degree)', 'log10(num individuals)')
    # Write data as JSON.
    if json_file:
        out_json_plot(json_file, range(len(values_cut)), values_cut,
                      description, 'Degree', 'Individuals')

# Calculates and plots shortest path lengths from a given node.
def distribution_shortest_path(node, graph, description, plot_on,
                               json_file=None):
    # Compute all shortest paths to that node.
    paths = nx.single_source_shortest_path(graph, node)
    # Place the lengths of those shortest paths into buckets.
    buckets = {}
    for node in paths:
        path = paths[node]
        path_len = len(path) - 1
        if path_len in buckets:
            buckets[path_len] += 1
        else:
            buckets[path_len] = 1
            
    # Output the histogram.
    print('Length \tCount')
    buckets_sorted = sorted([ int(key) for key in buckets.keys() ])
    values = [ buckets[bucket] for bucket in buckets_sorted ]
    print_table(buckets_sorted, values)
    # Also output the number of nodes that were not connected to the node
    # of highest degree.
    unconnected = len(graph.nodes()) - len(paths)
    print('Inf \t' + str(unconnected))
    # Plot the distribution.
    if plot_on:
        plot_distribution(buckets_sorted, values, description, 'Length',
                          'Individuals')
    # Write data as JSON.
    if json_file:
        out_json_plot(json_file, buckets_sorted, values, description,
                      'Length', 'Individuals')

# Print distribution of the lengths of the shortest paths from each node
# to the node of highest degree.
def distribution_shortest_path_rank(graph, plot_on, json_file=None):
    description = ('Distribution of lengths of shortest paths from node of ' +
                   'highest degree')
    print('\n' + description + ':')
    if len(graph.nodes()) == 0:
        return
    
    # Get node of highest degree.
    degr_nodes = graph.degree()
    node_highest_degree = sorted(degr_nodes, key=degr_nodes.get,
                                 reverse=True)[0]
    # Actually compute the shortest path histogram.
    distribution_shortest_path(node_highest_degree, graph, description,
                               plot_on, json_file=json_file)

# Print distribution of the lengths of the shortest paths from a specified
# source node.
def distribution_shortest_path_source(source, source_name, graph, plot_on,
                                      json_file=None):
    description = ('Distribution of lengths of shortest paths from ' +
                   str(source_name))
    print('\n' + description + ':')
    # Make sure the given individual is in the network.
    if not graph.has_node(source):
        print('ERROR: Could not find "' + source + '" in the network.')
        return
    # Actually compute the shortest path histogram.
    distribution_shortest_path(source, graph, description, plot_on,
                               json_file=json_file)

#################################
## YEARLY STATISTICS FUNCTIONS ##
#################################
    
# Print number of connected components for each year.
def year_stats_largest_component(years, plot_on=False, json_file=None,
                                 project='left'):
    description = 'Relative size of largest connected component'
    print('\n' + description + ':')

    largest = []
    second_largest = []
    for year in years:
        graph = load_graph(range(year-2, year+3), project)
        # If the graph has no nodes, then just append zeros and move on.
        if len(graph) == 0:
            largest.append(0)
            second_largest.append(0)
            continue
        # Sort the graph's connected components by number of nodes.
        sorted_cc = sorted(nx.connected_component_subgraphs(graph),
                           key=len, reverse=True)
        # Compute the fraction of nodes in largest CC.
        largest.append(len(sorted_cc[0]) / float(len(graph)))
        # Compute the fraction of nodes in second largest CC.
        if len(sorted_cc) == 1:
            second_largest.append(0.)
        else:
            second_largest.append(len(sorted_cc[1]) / float(len(graph)))

    print_table(years, largest, second_largest)
    if plot_on:
        plot_scatter(years, largest, description,
                    'Year', '%',
                    more=second_largest)
    if json_file:
        out_json_plot(json_file, years, largest, description,
                      'Year', '%')
    return largest, second_largest

# Plot change in degree over time for one or two individuals.
def year_stats_indiv_degree(indiv_1, indiv_2, years, plot_on=False,
                            project='left'):
    if not indiv_1 and not indiv_2:
        return
    description = 'Individual degree over time'
    print('\n' + description + ':')

    degrees_1 = []
    degrees_2 = []
    name_1 = ''
    name_2 = ''
    for year in years:
        graph = load_graph(range(year-2, year+3), project)
        if indiv_1:
            degree_1 = graph.degree(indiv_1, weight='weight')
            if degree_1 == {}:
                degree_1 = 0
            elif name_1 == '':
                name_1 = graph.node[indiv_1]['name']
            degrees_1.append(degree_1)
        if indiv_2:
            degree_2 = graph.degree(indiv_2)
            if degree_2 == {}:
                degree_2 = 0
            elif name_2 == '':
                name_2 = graph.node[indiv_2]['name']
            degrees_2.append(degree_2)

    if plot_on:
        plot_line(years, degrees_1, description, 'Year', 'Degree',
                  more=degrees_2)
        ax = plt.gca()
        legend_labels = [ name_1, name_2 ]
        ax.legend(legend_labels, loc='best')
    return degrees_1, degrees_2

# Print average shortest path length of the largest connected component
# for each year.
def year_stats_avg_short_path(years, plot_on=False, json_file=None,
                              project='left'):
    description = 'Average shortest path length over time'
    print('\n' + description + ':')
    aspl = []
    for year in years:
        graph = load_graph(range(year-2, year+3), project)
        largest_cc = max(nx.connected_component_subgraphs(graph),
                         key=len)
        if len(largest_cc.nodes()) <= 1:
            aspl.append(0.)
            continue
        aspl.append(nx.average_shortest_path_length(largest_cc))
            
    print_table(years, aspl)
    if plot_on:
        plot_scatter(years, aspl, description, 'Year',
                    'Average shortest path length')
    if json_file:
        out_json_plot(json_file, years, aspl, description, 'Year',
                      'Average shortest path length')
    return aspl

# Print average individual degree for each year.
def year_stats_avg_degree(years, plot_on, json_file=None,
                          project='left'):
    description = 'Average individual degree over time'
    print('\n' + description + ':')
    avg_degree = []
    for year in years:
        graph = load_graph(range(year-2, year+3), project)
        if len(graph.nodes()) == 0:
            avg_degree = append(0.)
        else:
            degr_nodes = graph.degree()
            degr_list = [ degr_nodes[n] for n in degr_nodes ]
            degr_sum = float(sum(degr_list))
            avg_degree.append(degr_sum / len(degr_list))
            
    print_table(years, avg_degree)
    if plot_on:
        plot_scatter(years, avg_degree, description, 'Year',
                    'Average degree')
    if json_file:
        out_json_plot(json_file, years, avg_degree, description, 'Year',                      
                      'Average degree')
    return avg_degree

# Print average clustering coefficient for each year.
def year_stats_avg_clustering(years, plot_on, json_file=None,
                              project='left'):
    description = 'Clustering coefficient over time'
    print('\n' + description +  ':')
    avg_clustering = []
    for year in years:
        graph = load_graph(range(year-2, year+3), project)
        if len(graph.nodes()) == 0:
            avg_clustering.append(0.)
        else:
            avg_clustering.append(nx.average_clustering(graph))
            
    print_table(years, avg_clustering)
    if plot_on:
        plot_scatter(years, avg_clustering, description, 'Year',
                    'Clustering coefficient')
    if json_file:
        out_json_plot(json_file, years, avg_clustering, description,
                      'Year', 'Clustering coefficient')
    return avg_clustering

###########################
## PAIR REPORT FUNCTIONS ##
###########################
        
# Report the shortest path between two individuals.
def pair_shortest_path(graph, id_1, id_2, name_1, name_2):
    description = 'Shortest path between two specified individuals'
    print('\n' + description + ':')

    if not graph.has_node(id_1):
        print('ERROR: Could not find "' + name_1 + '" in the network.')
        return
    if not graph.has_node(id_2):
        print('ERROR: Could not find "' + name_2 + '" in the network.')
        return

    path = nx.shortest_path(graph, source=id_1, target=id_2)
    for node in path:
        print(graph.node[node]['name'])

# Given two nodes in the same bipartite graph set, report all of the
# neighboring nodes in the other bipartite graph set.
def pair_common_neighbors(bigraph, id_1, id_2, name_1, name_2,
                          pub_graph=False, verbose=True):
    description = 'Neighbors shared between two specified nodes'
    if verbose:
        print('\n' + description + ':')

    if not bigraph.has_node(id_1):
        print('ERROR: Could not find "' + name_1 + '" in the network.')
        return
    if not bigraph.has_node(id_2):
        print('ERROR: Could not find "' + name_2 + '" in the network.')
        return

    neighbors_1 = set(bigraph.neighbors(id_1))
    neighbors_2 = set(bigraph.neighbors(id_2))
    for node in (neighbors_1 & neighbors_2):
        if pub_graph:
            print(bigraph.node[node]['name'])
        else:
            print(bigraph.node[node]['title'])

#####################################
## COMMAND LINE ARGUMENT FUNCTIONS ##
#####################################

# Given a string with an interval of years, return an integer list of all
# years within that inteveral (inclusive).
def year_interval(year_interval):
    split = year_interval.split('-')
    start = int(split[0].strip())
    if len(split) == 1:
        return [ start ]
    end = int(split[1].strip())
    # If the interval goes back in time, report an error.
    if end < start:
        raise argparse.ArgumentTypeError('Invalid time interval: "' +
                                         year_interval + '".')
    return range(start, end + 1)

# Check if given value is a non-negative integer.
def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(value + ' must be ' +
                                         'a non-negative integer.')
    return ivalue

# Open an output file.
def output_file(file_name):
    return open(file_name, 'w')

# Parse arguments.
def parse_args():
    parser = argparse.ArgumentParser(description='Report statistics '
                                     + 'on networks given in the specified '
                                     + 'time interval.')
    parser.add_argument('years', type=year_interval,
                        help='Either a single year (e.g., 1600) ' +
                        'or an interval of years delimited ' +
                        'by a dash (e.g., 1600-1610), over ' +
                        'which to load the graph data.')
    parser.add_argument('-i1', dest='indiv_1', default=None,
                        help='First individual; source of shortest ' + 
                        'path, path length distribution, and link ' +
                        'prediction analyses.')
    parser.add_argument('-i2', dest='indiv_2', default=None,
                        help='Second individual; source of shortest ' + 
                        'path, path length distribution, and link ' +
                        'prediction analyses.')
    parser.add_argument('--publications', dest='pub_analysis',
                        default=False, action='store_true',
                        help='Instead of reporting rankings and ' +
                        'distributions on individuals, report publications.')
    parser.add_argument('--json', dest='json_file',
                        default=None, type=output_file,
                        help='Specify an output file in which to save ' +
                        'the report in a JSON format. Default behavior ' +
                        'is not to write a JSON output.')
    parser.add_argument('--plot-off', dest='plot_off', default=False,
                        action='store_true',
                        help='Do not display plots of data generated ' +
                        'in this report.')
    parser.add_argument('--rank-off', dest='rank_off', default=False,
                        action='store_true',
                        help='Turn off the report that ranks ' +
                        'individuals based on certain metrics over ' + 
                        'the specified time interval.')
    parser.add_argument('--top-n', dest='top_n', default=10,
                        type=non_negative_int,
                        help='Specify the maximum number of ' +
                        'individuals to display in the ranking ' +
                        'report. By default, displays up to the ' +
                        'top 10 individuals.')
    parser.add_argument('--role', dest='role', default='all',
                        choices=[ 'author', 'printer', 'publisher',
                                  'bookseller', 'all' ],
                        help='If provided, then the ranking will be ' +
                        'filtered based on the individual\'s role.')
    parser.add_argument('--year-stats-off', dest='ys_off', default=False,
                        action='store_true',
                        help='Turn off the report that shows ' +
                        'different yearly network statistics over ' +
                        'the specified time interval.')
    parser.add_argument('--distr-off', dest='distr_off',
                        default=False, action='store_true',
                        help='Turn off the report that reports ' +
                        'distributions of various graph statistics.')
    parser.add_argument('--distr-cutoff', dest='distr_cutoff',
                        default=50, type=non_negative_int,
                        help='Only consider the degree distribution ' +
                        'until this value (default: 50)')
    return parser.parse_args()

##########
## MAIN ##
##########

if __name__ == '__main__':
    args = parse_args()

    plot_on = not args.plot_off

    # Load graph data over time interval.
    print('Loading network information from U Iowa database...')
    years = args.years
    bigraph = load_graph(years)
    print('Loading complete!')

    if args.pub_analysis:
        graph = project_graph(bigraph, 'right')
        if args.indiv_1:
            indiv_1 = int(args.indiv_1)
        if args.indiv_2:
            indiv_2 = int(args.indiv_2)
        if args.role != 'all':
            sys.stderr.write('WARNING: Cannot filter on role when ' +
                             'analyzing publications.\n')
            args.role = 'all'
    else:
        graph = project_graph(bigraph, 'left')
        indiv_1 = name_to_id(graph, args.indiv_1)
        indiv_2 = name_to_id(graph, args.indiv_2)

    # Ranking report.
    if args.rank_off:
        print('\nSkipping ranking report...')
    else:
        print('\n--------------------------')
        print('Individual ranking report:')
        print('--------------------------')
        if indiv_1:
            rank_link_prediction(indiv_1, graph, args.top_n,
                                 args.role, args.pub_analysis)
        if indiv_2:
            rank_link_prediction(indiv_2, graph, args.top_n,
                                 args.role, args.pub_analysis)
        rank_degree(graph, args.top_n, args.role, args.pub_analysis)
        rank_degree_weight(graph, args.top_n, args.role,
                           args.pub_analysis)
        rank_betweenness(graph, args.top_n, args.role, args.pub_analysis)
        rank_edge_betweenness(graph, bigraph, top_n=args.top_n,
                              pub_graph=args.pub_analysis,
                              plot_on=plot_on)

    # Distributions report.
    if args.distr_off:
        print('\nSkipping distributions report...')
    else:
        print('\n---------------------')
        print('Distributions report:')
        print('---------------------')
        distribution_degree(graph, args.distr_cutoff, plot_on,
                            json_file=args.json_file)
        #distribution_shortest_path_rank(graph, plot_on,
        #                                json_file=args.json_file)
        if indiv_1:
            distribution_shortest_path_source(indiv_1, args.indiv_1, graph,
                                              plot_on,
                                              json_file=args.json_file)
        if indiv_2:
            distribution_shortest_path_source(indiv_2, args.indiv_2, graph,
                                              plot_on,
                                              json_file=args.json_file)
        
    # Yearly statistics report.
    if args.ys_off:
        print('\nSkipping yearly statistics report...')
    else:
        print('\n-------------------------')
        print('Yearly statistics report:')
        print('-------------------------')
        if args.pub_analysis:
            project = 'right'
        else:
            project = 'left'
        year_stats_indiv_degree(indiv_1, indiv_2, years, plot_on, project)
        year_stats_largest_component(years, plot_on,
                                     args.json_file, project)
        year_stats_avg_degree(years, plot_on, args.json_file, project)
        year_stats_avg_clustering(years, plot_on,
                                  args.json_file, project)
        year_stats_avg_short_path(years, plot_on,
                                  args.json_file, project)
            
    # Pair of individuals report.
    if args.indiv_1 and args.indiv_2:
        print('\n------------')
        print('Pair report:')
        print('------------')
        pair_shortest_path(graph, indiv_1, indiv_2,
                           args.indiv_1, args.indiv_2)
        pair_common_neighbors(bigraph, indiv_1, indiv_2,
                              args.indiv_1, args.indiv_2,
                              args.pub_analysis)

    if args.json_file:
        args.json_file.close()
        
    if plot_on:
        plt.show()
