import re
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import sys
import json

bssid = sys.argv[1]
# bssid = "BSS00004"

mpl.style.use("default")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# This script removes redundant motifs from set of all motifs extracted by all models based on PWM similarity
# This script should be run after running "extract_CNN_CR_motifs.py"

# tomtom -no-ssc -verbosity 2 -text -dist pearson -evalue -thresh 5 /stg3/data1/sam/enhancer_prediction/gene_specific_motifs_3/genespecific_nonredundant_motifs_antiCISBP_enh.meme /stg3/data1/sam/enhancer_prediction/motif_analysis/Epigram_human.meme > /stg3/data1/sam/enhancer_prediction/motif_analysis/enh3_antiCISBP_overlap_Epigram_motifs.txt

# Folder and constants
os.chdir(os.path.dirname(os.path.abspath(__file__)))
REMARK = "256MOTIF_pytorch"
type = sys.argv[2] if len(sys.argv) > 2 else "pro"   # window type: pro or enh

def get_meme_size(meme_file):
    motif_names = []
    with open(meme_file, 'r') as file:
        for line in file:
            if line.startswith('MOTIF'):
                motif_name = line.strip().split(' ')[1]
                motif_names.append(motif_name)
    return len(motif_names)

def get_consensus_sequence(pwm):
    """Extracts the consensus sequence from a PWM."""
    # Define the nucleotides corresponding to each column
    nucleotides = 'ACGT'
    # Find the index of the maximum value in each row (position)
    max_indices = np.argmax(pwm, axis=1)
    # Map indices to nucleotides to form the consensus sequence
    consensus = ''.join(nucleotides[idx] for idx in max_indices)
    return consensus

def reverse_complement_seq(sequence):
    """
    Compute the reverse complement of a DNA sequence.

    Args:
        sequence (str): Input DNA sequence (e.g., "ACGT").

    Returns:
        str: Reverse complement of the input sequence.
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement[base] for base in reversed(sequence))

def reverse_complement_pwm(pwm):
    """
    Compute the reverse complement of a PWM.
    Reverse the rows and swap columns: A ↔ T and C ↔ G.

    Args:
        pwm (np.array): PWM of shape (L, 4).

    Returns:
        np.array: Reverse complement of the PWM.
    """
    complement_map = [3, 2, 1, 0]  # A=0, C=1, G=2, T=3
    return pwm[::-1, :][:, complement_map]

def calculate_information_content(pwm, background_freq=0.25):
    ic = 0
    for position in pwm:
        for prob in position:
            if prob > 0:
                ic += prob * np.log2(prob / background_freq)
    return ic

def compare_motifs(motif1, motif2):
    """
    Compares the consensus sequences of two PWM motifs.
    
    Args:
    motif1 (np.array): An 8x4 matrix representing the first motif's PWM.
    motif2 (np.array): An 8x4 matrix representing the second motif's PWM.
    
    Returns:
    bool: True if the consensus sequences are the same, False otherwise.
    """
    # Get the consensus sequences of both motifs
    consensus1 = get_consensus_sequence(motif1)
    consensus2 = get_consensus_sequence(motif2)
    
    # Compare the consensus sequences
    return consensus1 == consensus2

def find_common_motifs(list_of_motif_lists, list_of_weight_lists, shared):
    """
    Finds and combines motifs that appear in more than 'shared' different lists by their consensus sequences,
    while extending the corresponding weights array to include all weights from the 10 lists.

    Args:
    list_of_motif_lists (list): A list of lists, each containing arrays of PWMs (motifs).
    list_of_weight_lists (list): A list containing lists, each with arrays of weights corresponding to each motif.
    shared (int): The threshold for the number of lists a motif must appear in to be considered common.

    Returns:
    tuple: Two lists - one containing the combined motifs, and the other containing the extended weight arrays.
    """
    consensus_dict = {}

    # Assign consensus sequences to all motifs and group motifs (and their weights) by these sequences
    for motif_list, weight_list in zip(list_of_motif_lists, list_of_weight_lists):
        for motif_array, weight_array in zip(motif_list, weight_list):
            # Calculate consensus sequence for the motif array
            consensus = get_consensus_sequence(motif_array[0])  # All motifs in the array share the same consensus
            
            if consensus not in consensus_dict:
                # Create a new entry with the current array of motifs and weights
                consensus_dict[consensus] = {'motifs': [motif_array], 'weights': [weight_array]}
            else:
                # Append the current array of motifs and weights to the existing entry
                consensus_dict[consensus]['motifs'].append(motif_array)
                consensus_dict[consensus]['weights'].append(weight_array)


    # Filter and combine motifs that occur in more than 'shared' different lists
    combined_motifs = []
    extended_weights = []
    
    for consensus, data in consensus_dict.items():
        motifs = data['motifs']
        weights = data['weights']
        
        if len(motifs) >= shared:  # Check if there are more than 'shared' motifs with the same consensus
            # Combine motifs to find the most representative motif
            combined_motif_array = np.vstack(motifs)
            combined_motif = get_representative_motif(combined_motif_array)

            # Extend weights by concatenating all weights corresponding to this consensus motif
            extended_weight_array = np.concatenate(weights)

            # Append combined motifs and extended weights to separate lists
            combined_motifs.append(combined_motif)
            extended_weights.append(extended_weight_array)

    # Return two separate lists: one for motifs and one for extended weights
    return combined_motifs, extended_weights

def calculate_pearson_distance(pwm1, pwm2):
    """
    Calculate the Pearson distance between two PWMs.

    Args:
        pwm1, pwm2 (np.array): PWMs of shape (L, 4).

    Returns:
        float: Pearson distance (1 - correlation).
    """
    L = pwm1.shape[0]  # Length of the PWM
    distances = []
    for i in range(L):
        correlation = np.corrcoef(pwm1[i], pwm2[i])[0, 1]
        distances.append(1 - correlation)  # Pearson distance
    return np.mean(distances)

def write_motifs_to_meme_file(motifs, file_path, motif_names=None):
    with open(file_path, 'w') as f:
        f.write('MEME version 4.1.1\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        for i, motif in enumerate(motifs):
            motif_name = motif_names[i]
            f.write('MOTIF {0} {1}\n'.format(i, motif_name))
            f.write('letter-probability matrix: alength=4 w={0} nsites=1 E=0.0e+00\n'.format(motif.shape[0]))
            for row in motif:
                f.write('\t'.join([str(x) for x in row]) + '\n')
            f.write('\n')

def parse_meme_to_pwms(meme_file):
    """
    Parse a MEME file and convert the PWMs into a list of 2D NumPy arrays.
    """
    with open(meme_file, 'r') as file:
        lines = file.readlines()

    motifs = []
    reading_pwm = False
    current_pwm = []

    for line in lines:
        # Check for the start of a PWM section
        if line.startswith('MOTIF'):
            if current_pwm:
                # Convert the current PWM to a NumPy array and add to motifs list
                motifs.append(np.array(current_pwm))
                current_pwm = []
            reading_pwm = False

        # Check for the start of the actual PWM data
        elif line.startswith('letter-probability matrix'):
            reading_pwm = True
            current_pwm = []
            continue

        # Read and store the PWM data
        if reading_pwm and line.strip():
            pwm_row = [float(x) for x in line.strip().split()]
            current_pwm.append(pwm_row)

    # Add the last PWM if present
    if current_pwm:
        motifs.append(np.array(current_pwm))

    return motifs

def read_tomtom_results(tomtom_tsv, q_value_threshold, E_value_threshold, p_value_threshold):
    """
    Read TOMTOM TSV results and filter out reverse complementaries.
    """
    if not os.path.exists(tomtom_tsv) and os.path.exists(tomtom_tsv + '.gz'): tomtom_tsv = tomtom_tsv + '.gz'   # the shipped tables are gzipped
    tomtom_results = pd.read_csv(tomtom_tsv, sep='\t', comment='#')
    tomtom_results['Query_ID'] = tomtom_results['Query_ID'].astype(int)
    tomtom_results['Target_ID'] = tomtom_results['Target_ID'].astype(int)
    tomtom_results['Optimal_offset'] = tomtom_results['Optimal_offset'].astype(int)

    unique_values_count = tomtom_results['Query_ID'].nunique()
    # print("Number of unique nodes:", unique_values_count)

    # Filter the DataFrame and convert columns to integers

    tomtom_results = tomtom_results[(tomtom_results['Optimal_offset'] == 0) & 
                                    (tomtom_results['E-value'] <= E_value_threshold) & 
                                    (tomtom_results['p-value'] <= p_value_threshold)]
    
    # Filter to keep rows where column_a and column_b have exactly the same string value
    # tomtom_results = tomtom_results[(tomtom_results['Query_consensus'] == tomtom_results['Target_consensus']) &
    #                                 (tomtom_results['Orientation'] == '+')]
    return tomtom_results

def calculate_relative_offsets(group, tomtom_results):
    """
    Calculate the relative offsets of each motif in the group with respect to a reference motif.
    """
    # Choose the first motif in the group as a reference
    reference_motif = min(group)
    relative_offsets = {motif: 0 for motif in group}
    
    for motif in group:
        if motif == reference_motif:
            continue
        # Sum the offsets from the reference to the current motif
        offset = tomtom_results[((tomtom_results['Query_ID'] == reference_motif) & 
                                (tomtom_results['Target_ID'] == motif)) |
                               ((tomtom_results['Query_ID'] == motif) & 
                                (tomtom_results['Target_ID'] == reference_motif))]['Optimal_offset'].sum()
        relative_offsets[motif] = offset
    
    return relative_offsets

def adjust_and_combine_motifs(motif_group, motifs, tomtom_results):
    """
    Adjust motifs by their relative offsets and combine them by taking the average.
    Ensure that the combined PWM sums up to 1 at each position.
    """
    relative_offsets = calculate_relative_offsets(motif_group, tomtom_results)

    # Determine the maximum necessary padding for the motifs
    max_length = max(motifs[motif_id].shape[0] + abs(offset) 
                     for motif_id, offset in relative_offsets.items())

    # Initialize arrays to accumulate combined motifs
    combined_motif = np.zeros((max_length, 4))

    # Adjust each motif by its relative offset and combine
    for motif_id, offset in relative_offsets.items():
        motif = motifs[motif_id]
        adjusted_motif = np.pad(motif, ((abs(offset), max_length - motif.shape[0] - abs(offset)), (0, 0)), 'constant')
        combined_motif += adjusted_motif

    # Normalize the combined motif at each position
    combined_motif = np.divide(combined_motif, np.sum(combined_motif, axis=1, keepdims=True), out=np.zeros_like(combined_motif), where=np.sum(combined_motif, axis=1, keepdims=True)!=0)

    return combined_motif

def get_representative_motif(motifs):
    """
    Select the most representative motif by finding the motif closest to the component-wise
    median motif using Pearson distance.

    Parameters:
    motifs (list or np.array): List or array of motifs, where each motif is of shape (L, 4).

    Returns:
    np.array: The most representative motif.
    """
    # if only 1 motif, return this motif
    if len(motifs) == 1:
        return motifs[0]
    
    # Stack motifs into a single array of shape (num_motifs, L, 4)
    motif_stack = np.stack(motifs, axis=0)

    # Step 1: Compute the component-wise median motif
    median_motif = np.median(motif_stack, axis=0)  # Shape (L, 4)

    # Step 2: Find the motif closest to the median motif using Pearson distance
    distances = [calculate_pearson_distance(median_motif, motif) for motif in motifs]
    closest_index = np.argmin(distances)  # Find the index of the closest motif

    # Return the most representative motif
    representative_motif = motifs[closest_index]
    return representative_motif

def combine_motifs_by_network(motifs, tomtom_results, weights, threshold):
    """
    Get combined motifs from TOMTOM results, considering cumulative distance
    in the similarity network, ensuring the maximum cumulative distance in each group
    is less than the threshold.

    Args:
        motifs (list): List of motifs.
        tomtom_results (DataFrame): TOMTOM results containing Query_ID, Target_ID, and E-value.
        weights (array): Array of weights corresponding to motifs.
        threshold (float): Maximum cumulative distance to allow within a group.

    Returns:
        tuple: Lists of grouped motifs and their weights.
    """
    G = build_similarity_network(tomtom_results)

    # Compute shortest path distances between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))
    # print node 519 shortest path neighbours
    # print(shortest_paths[519])

    # Initialize clusters and visited set
    clusters = []
    visited = set()

    # Iteratively form clusters
    for node in G.nodes:
        if node in visited:
            continue

        # Start a new cluster with the current node
        cluster = set([node])
        visited.add(node)
        to_visit = [node]

        while to_visit:
            current = to_visit.pop()
            for neighbor, distance in shortest_paths[current].items():
                if neighbor not in visited:
                    # Check if adding this neighbor satisfies the threshold condition
                    if all(
                        shortest_paths[member].get(neighbor, float('inf')) <= threshold
                        for member in cluster
                    ):
                        cluster.add(neighbor)
                        visited.add(neighbor)
                        to_visit.append(neighbor)

        # Add the valid cluster to the list of clusters
        clusters.append(cluster)

    group_motifs = []
    group_weights = []
    group_id = []

    for cluster in clusters:
        group = np.array([motif_id for motif_id in cluster])
        curr_group_weights = weights[group]
        group_weights.append(curr_group_weights)
        relevant_motifs = [motifs[i] for i in group]
        # combined_motif = adjust_and_combine_motifs(group, motifs, tomtom_results)
        # combined_motif = get_representative_motif(relevant_motifs)
        group_motifs.append(relevant_motifs)
        group_id.append(group)
    
    return group_motifs, group_weights, group_id

def combine_motifs_clique_network(motifs, tomtom_results, weights, threshold):
    """
    Get combined motifs from TOMTOM results, ensuring each cluster forms a fully connected subgraph.
    Only include motifs where all nodes in the cluster are connected to one another.

    Args:
        motifs (list): List of motifs.
        tomtom_results (DataFrame): TOMTOM results containing Query_ID, Target_ID, and E-value.
        weights (array): Array of weights corresponding to motifs.
        threshold (float): Maximum cumulative distance to allow within a group.

    Returns:
        tuple: Lists of grouped motifs, their weights, and their IDs.
    """
    G = build_similarity_network(tomtom_results)

    # # Filter edges based on the threshold to ensure all connections are within the threshold
    # edges_within_threshold = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] <= threshold]
    # G = nx.Graph()
    # G.add_weighted_edges_from(edges_within_threshold)

    # Initialize clusters and visited set
    clusters = []
    visited = set()

    # Iterate over connected components and split into fully connected subgraphs
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # Process cliques (fully connected subgraphs)
        cliques = list(nx.find_cliques(subgraph))  # Find all maximal cliques
        for clique in cliques:
            # Ensure all nodes in the clique are not already visited
            clique_set = set(clique)
            if not clique_set & visited:  # Only consider nodes not in the visited set
                clusters.append(clique_set)
                visited.update(clique_set)  # Mark nodes as visited

    # Handle isolated nodes
    for node in G.nodes:
        if node not in visited:
            clusters.append({node})
            visited.add(node)

    # Prepare grouped motifs, weights, and IDs
    group_motifs = []
    group_weights = []
    group_id = []

    for cluster in clusters:
        group = np.array([motif_id for motif_id in cluster])
        curr_group_weights = weights[group]
        group_weights.append(curr_group_weights)
        relevant_motifs = [motifs[i] for i in group]
        # combined_motif = adjust_and_combine_motifs(group, motifs, tomtom_results)
        # combined_motif = get_representative_motif(relevant_motifs)
        group_motifs.append(relevant_motifs)
        group_id.append(group)

    return group_motifs, group_weights, group_id

def build_similarity_network(tomtom_results):
    """
    Build a weighted network of motifs based on the TOMTOM similarity results.
    The edge weight is determined by the E-value.
    """
    # Build the network graph
    G = nx.Graph()
    for _, row in tomtom_results.iterrows():
        # Add an edge with the E-value as the weight
        G.add_edge(row['Query_ID'], row['Target_ID'], weight=row['E-value'])
    return G

def deduplicate_reverse_complements(rep_motifs):
    """
    Remove reverse complementary motifs from a list of PWMs.

    Args:
        rep_motifs (list): List of PWM motifs.

    Returns:
        list: Deduplicated motifs with reverse complements removed.
    """
    # Step 1: Compute consensus sequences
    consensus_to_motif = {}
    for motif in rep_motifs:
        consensus = get_consensus_sequence(motif)  # Compute the consensus sequence
        rc_seq = reverse_complement_seq(consensus)  # Compute reverse complement

        # Step 2: Deduplicate reverse complements
        if consensus not in consensus_to_motif and rc_seq not in consensus_to_motif:
            consensus_to_motif[consensus] = motif

    # Step 3: Return the deduplicated motifs
    return list(consensus_to_motif.values())

def normalize_motifs(motifs):
    """
    Normalize motifs by selecting the orientation (normal or reverse complement)
    that is most similar to the motif with the highest information content.

    Args:
        motifs (list of np.array): List of PWMs.

    Returns:
        list of np.array: Normalized motifs.
    """

    # if motifs length is 1 return 
    if len(motifs) == 1:
        return motifs
    
    # Find the motif with the highest information content
    ic_scores = [calculate_information_content(pwm) for pwm in motifs]
    reference_index = np.argmax(ic_scores)  # Index of the highest IC motif
    reference_pwm = motifs[reference_index]

    # Normalize all motifs relative to the reference
    normalized_motifs = []
    for pwm in motifs:
        rc_pwm = reverse_complement_pwm(pwm)

        # Compare Pearson distance to the reference motif
        normal_dist = calculate_pearson_distance(reference_pwm, pwm)
        reverse_dist = calculate_pearson_distance(reference_pwm, rc_pwm)

        # Choose the orientation with the smaller Pearson distance
        normalized_pwm = pwm if normal_dist <= reverse_dist else rc_pwm
        normalized_motifs.append(normalized_pwm)

    return normalized_motifs

celldir = (sys.argv[3].rstrip('/') + '/') if len(sys.argv) > 3 else '/stg3/data1/sam/enhancer_prediction/cell_type_specific_CNN/' + bssid + '/'   # optional third argument: folder holding the motif files
shared_motifs = []
shared_weights = []
# Add clustering tracking variables
all_clustering_info = []  # List to store clustering info for each iteration
motif_names = []  # List to store motif names for final output

# Go over all 10 models (iterations)
for iteration in range(10):
    full_model_name = bssid + "_it_" + str(iteration) + "_" + REMARK
    print("Loading model " + full_model_name)

    meme_file = celldir + full_model_name + '_' + str(iteration) + f'_{type}_motifs_all_256.meme'
    motifs = parse_meme_to_pwms(meme_file)
    weights = np.load(celldir + full_model_name + '_' + str(iteration) + f'_{type}_motifs_weights_all_256.npy')
    # Calculate IC for each PWM
    ic_values = [calculate_information_content(pwm) for pwm in motifs]

    # # Sort each array from largest to smallest
    # sorted_arrays = np.sort(weights)[::-1]

    # # Plot each sorted array as a line
    # plt.figure(figsize=(5, 3))
    # plt.plot(sorted_arrays, label=f"Array")

    # # Add labels, legend, and title
    # plt.title("Sorted Arrays as Lines")
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig(f"./tmp2.pdf", dpi=300, bbox_inches="tight", transparent=True)
    # plt.close()

    # continue

    # Final lists to hold top 750 motifs and their weights
    sorted_indices = np.argsort(weights)[::-1]
    selected_indices = sorted_indices[:750]

    # selected_indices = []
    # all_weight = []
    # for i in range(10):
    #     # Define the start and end indices for the current partition
    #     start = i * 256
    #     end = start + 256
        
    #     # Get the weights for the current partition
    #     partition_weights = weights[start:end]
    #     all_weight.append(partition_weights)
        
    #     # Sort the weights by descending order and get the indices
    #     sorted_indices = np.argsort(partition_weights)[::-1]
        
    #     # Select the top 30 indices
    #     if i == 0:
    #         top_indices = sorted_indices[:256]
    #     else:
    #         top_indices = sorted_indices[:50]
        
    #     # Adjust indices to global range and add to the selected list
    #     selected_indices.extend(start + idx for idx in top_indices)
    
    # # Sort each array from largest to smallest
    # sorted_arrays = [np.sort(arr)[::-1] for arr in all_weight]

    # # Plot each sorted array as a line
    # plt.figure(figsize=(8, 5))
    # for i, sorted_array in enumerate(sorted_arrays):
    #     plt.plot(sorted_array, label=f"Branch {i+1}")

    # # Add labels, legend, and title
    # plt.title("GM12878 Model 1")
    # plt.xlabel("Sorted Motif Index")
    # plt.ylabel("Contextual Weight")
    # plt.legend()
    # plt.savefig(f"./tmp2.png", dpi=300, bbox_inches="tight", transparent=True)
    # plt.close()

    # continue

    # Filter for high IC motifs
    selected_motifs = [motifs[idx] for idx in selected_indices]
    ic_values_selected = [calculate_information_content(pwm) for pwm in selected_motifs]
    filtered_ic_selected = [ic > 10 for ic in ic_values_selected]
    selected_indices = [idx for idx, flag in zip(selected_indices, filtered_ic_selected) if flag]

    # # Plot a histogram of the IC values
    # plt.figure(figsize=(4, 2))
    # plt.hist(ic_values, bins=30, color='#D3D3D3', edgecolor='black', linewidth=0.5)
    # plt.title('IC Distribution of All Motifs in GM12878 Model 1')
    # plt.xlabel('Information Content (IC)')
    # plt.ylabel('Frequency')
    # plt.savefig("./ic.png", dpi=300, bbox_inches="tight", transparent=True)
    # plt.close()
    # # Plot a histogram of the IC values of top 750 motifs
    # plt.figure(figsize=(4, 2))
    # plt.hist(ic_values_selected, bins=30, color='#D3D3D3', edgecolor='black', linewidth=0.5)
    # plt.title('IC Distribution of Top 750 Motifs in GM12878 Model 1')
    # plt.xlabel('Information Content (IC)')
    # plt.ylabel('Frequency')
    # plt.savefig("./ic_selected.png", dpi=300, bbox_inches="tight", transparent=True)
    # plt.close()

    # Read TOMTOM results
    tomtom_tsv = celldir + full_model_name + f'_{type}_motifs_all_tomtom.txt'  # Path to your TOMTOM TSV file
    tomtom_results = read_tomtom_results(tomtom_tsv, q_value_threshold=0.05, E_value_threshold=1, p_value_threshold=0.05)
    tomtom_results = tomtom_results[tomtom_results['Query_ID'].isin(selected_indices) & tomtom_results['Target_ID'].isin(selected_indices)]

    # Get combined motifs considering similarity network
    group_motifs, group_weights, group_ids = combine_motifs_by_network(motifs, tomtom_results, weights, 1)
    print(len(np.concatenate(group_motifs)))
    print(len(group_motifs))

    # Save clustering information for this iteration
    iteration_clustering_info = {
        'iteration': iteration,
        'groups': []
    }
    
    for group_idx, group_id in enumerate(group_ids):
        group_info = {
            'group_name': f'it{iteration}group{group_idx}',
            'original_motif_indices': group_id.tolist()  # Convert numpy array to list for JSON serialization
        }
        iteration_clustering_info['groups'].append(group_info)
    
    all_clustering_info.append(iteration_clustering_info)

    # Normalize motifs in groups
    group_motifs = [normalize_motifs(motif_group) for motif_group in group_motifs]

    # iterate over each group of motifs
    for group_idx, (group_motif, group_weight) in enumerate(zip(group_motifs, group_weights)):
        shared_motifs.append(get_representative_motif(group_motif))
        shared_weights.append(group_weight)
        # Add motif name for tracking
        motif_names.append(f'it{iteration}group{group_idx}')

# Combine motifs from all models
write_motifs_to_meme_file(shared_motifs, celldir + f'{bssid}_{type}_motifs_combined.meme', motif_names)
np.savez(celldir + f'{bssid}_{type}_motifs_combined_weights.npz', *shared_weights)

# Save clustering information to JSON file
with open(celldir + f'{bssid}_{type}_clustering_info_1.json', 'w') as f:
    json.dump(all_clustering_info, f)



# median_motif_weight = np.array([np.median(arr) for arr in shared_weights_reduced])

# # Take top jaspar number of motifs by median group weight
# num_jaspar = get_meme_size('/stg3/data1/sam/enhancer_prediction/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme')
# num_motifs = min(num_jaspar, len(median_motif_weight))
# num_motifs = len(median_motif_weight)
# sorted_median_weights = np.argsort(median_motif_weight)[::-1]
# top_weight_indices = sorted_median_weights[:num_motifs]
# top_motifs = [shared_motifs_reduced[idx] for idx in top_weight_indices]
# top_weights = [shared_weights_reduced[idx] for idx in top_weight_indices]
# top_weights_median = median_motif_weight[top_weight_indices]

# write_motifs_to_meme_file(shared_motifs_reduced, celldir + bssid + f'_{type}_motifs_nonredundant.meme')
# np.save(celldir + bssid + f'_{type}_motifs_nonredundant_weights.npy', median_motif_weight)
# #save motif weight as txt file 1 weight per line
# np.savetxt('./motifs_nonredundant_weights.txt', top_weights_median, fmt='%f')


# # Create the KDE plot
# sorted_weights = np.sort(median_motif_weight)[::-1]
# sns.set(style="whitegrid")  # for a nice grid background
# plt.figure(figsize=(6, 3))  # adjust figure size
# plt.plot(sorted_weights, color='royalblue')  # plot smooth curve
# plt.fill_between(np.arange(len(sorted_weights)), sorted_weights, color='royalblue', alpha=0.1)
# plt.xlabel('Motif')  # set x-label
# plt.ylabel('Contextual Weight Sum')  # set y-label
# plt.savefig(f"./combine_weights_{type}.pdf", dpi=300, bbox_inches="tight", transparent=True)
# plt.close()

# # Filter out arrays with only length 1
# shared_weights_reduced = [arr for arr in shared_weights_reduced if len(arr) > 2]

# # Plot heat map
# global_min = min(arr.min() for arr in shared_weights_reduced)
# global_max = max(arr[arr > 0].max() if np.any(arr > 0) else 0 for arr in shared_weights_reduced)
# # positive_percentiles = np.linspace(0, global_max, 4)  # 0 to max, split into 3 intervals
# # percentiles = np.linspace(global_min, global_max, 10)  # 0 to max, split into 3 interval
# # percentiles = np.concatenate(([global_min], positive_percentiles))

# # Non-linear scaling: quadratic bias towards smaller values
# num_points = 10
# linear_space = np.linspace(0, 1, num_points)  # Uniform spacing in [0, 1]
# non_linear_space = linear_space**2  # Apply quadratic transformation
# percentiles = global_min + non_linear_space * (global_max - global_min)

# # Classify weights into bins and calculate percentages
# percentages = []  # To hold the percentage of weights in each bin for all motif groups
# for arr in shared_weights_reduced:
#     bin_counts = np.digitize(arr, percentiles, right=True)
#     bin_percentages = [np.sum(bin_counts == i) / len(arr) * 100 for i in range(1, len(percentiles))]
#     percentages.append(bin_percentages)
# # Create the heatmap data frame with motif groups as rows (which will be transposed to columns)
# heatmap_df = pd.DataFrame(percentages).T  # Transpose at creation
# # Naming the rows (which will become columns after transposing again for visualization)
# heatmap_df.index = [f'{percentiles[i]:.4f}-{percentiles[i+1]:.4f}' for i in range(len(percentiles)-1)]
# # Naming the columns (which represent the groups, and are correct as rows before transposing for visualization)
# heatmap_df.columns = [f'Group {i+1}' for i in range(len(percentages))]

# # Plotting the heatmap
# plt.figure(figsize=(15, 2))  # Adjust the figure size to ensure the heatmap is horizontally elongated
# sns.heatmap(heatmap_df, cmap='coolwarm', annot=False, cbar_kws={'label': 'Percentage'})
# plt.title('Distribution of Motif Group Weights Across Percentiles')
# plt.ylabel('Percentile Ranges')
# plt.xlabel('Motif Groups')
# plt.xticks([])
# plt.savefig(f"./combine_motifs_{type}.pdf", dpi=300, bbox_inches="tight", transparent=True)
# plt.close()

# # Plot standard deviation distribution
# std_devs = np.array([np.std(arr, ddof=1) for arr in top_weights if len(arr) >= 2])
# # save std_dev as np array
# np.save(f'./{type}_motifs_nonredundant_stdDev.npy', std_devs)

# # Read std_dev promoter
# std_devs_pro = np.load(f'./pro_motifs_nonredundant_stdDev.npy')
# # Read std_dev enhancer
# std_devs_enh = np.load(f'./enh_motifs_nonredundant_stdDev.npy')
# # remove nan values
# std_devs_pro = std_devs_pro[~np.isnan(std_devs_pro)]
# std_devs_enh = std_devs_enh[~np.isnan(std_devs_enh)]

# # Normalize using min-max scaling
# def min_max_normalize(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

# std_devs_pro_norm = min_max_normalize(std_devs_pro)
# std_devs_enh_norm = min_max_normalize(std_devs_enh)

# # Create a new figure
# plt.figure(figsize=(4, 1.5))

# # Plot density plots
# sns.kdeplot(std_devs_pro_norm, shade=True, color="r", label="Promoter")
# sns.kdeplot(std_devs_enh_norm, shade=True, color="b", label="Enhancer")

# # Set plot labels and title
# plt.xlabel('Normalized Std', fontsize=10)
# plt.ylabel('Density', fontsize=10)
# # Set x lim
# plt.xlim(0, 1)

# # Display legend
# plt.legend(fontsize=10)
# plt.savefig("./variation_distribution.pdf", dpi=300, bbox_inches='tight', transparent=True)
# plt.close()    
    
    
# plt.figure(figsize=(6, 3))  # adjust figure size


# color = '#D1B2B1'
# # Plot the fill (shade) with transparency
# sns.kdeplot(std_devs, fill=True, color=color, alpha=0.2)
# # Plot the line on top of the filled area
# sns.kdeplot(std_devs, color=color)
# plt.xlabel('Motif Weight Std. Deviation')  # set x-label
# plt.ylabel('Density')  # set y-label
# plt.savefig(f"./motif_weight_stdDeviation_{type}.pdf", dpi=300, bbox_inches="tight", transparent=True)
# plt.close()


print("Done")
