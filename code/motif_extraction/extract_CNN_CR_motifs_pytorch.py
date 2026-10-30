import sys
# Import packages
import os
import numpy as np
from math import *
import random
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss

# Maintain tf 1.x behaviour under tf 2.x
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed_all(42)
    CUDA = True
else:
    print("CUDA is not available.")
    CUDA = False

# Constants
NUM_ACTIVATIONS = 1000
NUM_BRANCHES = 10

# Paths
TRAINED_MODELS = "./trained_models/"
MODEL_METADATA = "./trained_models_data/"
saveXdir=ENHANCER_ROOT + "/training_one_hot"

PRO_ONLY = False

NUM_MOTIFS = 256
MOTIF_LENGTH = 8

MAXPOOL0_SIZE = 5
MAXPOOL1_SIZE = 2
MAXPOOL2_SIZE = 3
MAXPOOL3_SIZE = 4
CONV1_FILTER = 64
CONV2_FILTER = 64
CONV3_FILTER = 64

REG1 = 0.1
REG2 = 0.2
REG3 = 0.3
DROPOUT1 = 0.15
DROPOUT2 = 0.15

batch = 64
epoch = 100
lr = 0.0001
decay = 0
patience = 10

# X and y data directories
saveXdir=ENHANCER_ROOT + "/training_one_hot"
saveydir=ENHANCER_ROOT + "/training_y"

REMARK = "256MOTIF_pytorch"
GENES = 18356
# Shuffle input genes order
random_genes = list(range(GENES))

if PRO_ONLY:
    NUM_BRANCHES = 10
else:
    NUM_BRANCHES = 20

# Use only non redundant cell lines to train & filter out cells with bad QC
nonDupBSSID = np.genfromtxt(ENHANCER_ROOT + "/Epimap_BSSID_nondup.txt", delimiter="\t", dtype=str)
badQC = ["BSS01297", "BSS01325"]
nonDupBSSID = np.array([item for item in nonDupBSSID if item not in badQC])
nonDupBSSID = sys.argv[1:] or list(nonDupBSSID)   # biosample(s) on the command line; default: all
# nonDupBSSID = sys.argv[1:] or list(nonDupBSSID)   # biosample(s) on the command line; default: all


# Custome bias layer
class BiasLayer(nn.Module):
    def __init__(self, num_features):
        super(BiasLayer, self).__init__()
        # self.bias = Parameter(torch.zeros(num_features))  # Bias parameter
        self.weight = Parameter(torch.ones(num_features))  # Weight parameter

    def forward(self, x):
        return x * self.weight
        # return x + self.bias
        # return (x + self.bias) * self.weight

class MultiBranchModel(nn.Module):
    def __init__(self, num_branches=20):
        super(MultiBranchModel, self).__init__()
        # Assuming each branch takes input of shape (400, 4)
        self.branches = nn.ModuleList([CNNBranch(branch_number=i) for i in range(num_branches)])

    def forward(self, x):
        # Convert x to float, if not already
        x = x.float()
        # x is expected to have shape [batch_size, 20, 400, 4]
        batch_size = x.size(0)
        
        # Initialize outputs and mask for non-zero input tracking
        branch_outputs = torch.zeros(batch_size, len(self.branches), device=x.device)
        bias_outputs = []
        contextual_outputs = []
        combined_outputs = []

        for i, branch in enumerate(self.branches):
            branch_input = x[:, i, :, :]
            branch_output, contextual_output, combined_output, bias_output = branch(branch_input)
            branch_outputs[:, i] = branch_output.squeeze()
            # print(bias_output.shape)
            bias_outputs.append(bias_output.squeeze())
            contextual_outputs.append(contextual_output.squeeze())
            combined_outputs.append(combined_output.squeeze())

        final_output = branch_outputs.sum(dim=1, keepdim=True)

        return final_output, torch.stack(contextual_outputs, dim=1), torch.stack(combined_outputs, dim=1), torch.stack(bias_outputs, dim=0)
    
class CNNBranch(nn.Module):
    def __init__(self, branch_number):
        super(CNNBranch, self).__init__()
        self.branch_number = branch_number
        
        # Initial Convolution
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=NUM_MOTIFS, kernel_size=MOTIF_LENGTH, padding='valid', bias=False)   
        init.xavier_normal_(self.conv0.weight)
        # init.kaiming_normal_(self.conv0.weight, nonlinearity='relu')


        # Continuing the main path
        self.maxpool0 = nn.MaxPool1d(MAXPOOL0_SIZE)
        
        self.conv1 = nn.Conv1d(in_channels=NUM_MOTIFS, out_channels=CONV1_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv1.weight)
        # init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.maxpool1 = nn.MaxPool1d(MAXPOOL1_SIZE)
        
        self.conv2 = nn.Conv1d(in_channels=CONV1_FILTER, out_channels=CONV2_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv2.weight)
        # init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.maxpool2 = nn.MaxPool1d(MAXPOOL2_SIZE)
        
        self.conv3 = nn.Conv1d(in_channels=CONV2_FILTER, out_channels=CONV3_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv3.weight)
        # init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.maxpool3 = nn.MaxPool1d(MAXPOOL3_SIZE)
        
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(192, NUM_MOTIFS) 
        self.dropout1 = nn.Dropout(DROPOUT1)
        
        self.dense2 = nn.Linear(NUM_MOTIFS, NUM_MOTIFS)
        self.dropout2 = nn.Dropout(DROPOUT2)

        self.contextual_weight = nn.Linear(NUM_MOTIFS, NUM_MOTIFS)
        self.bias_layer = BiasLayer(1)
    
    def forward(self, x):

        # is_non_zero_sequence = torch.any(x.view(x.size(0), -1) != 0, dim=1).unsqueeze(-1).float()
        is_non_zero_sequence = torch.any(x[:, 0, :] != 0, dim=1).unsqueeze(-1).float()

        x = x.permute(0, 2, 1)
        # Initial motif detection
        x = self.conv0(x)
        # x = torch.exp(x)
        # x = F.relu(x)

        # Get mean motif activation
        mean_activation = torch.mean(x, dim=2)
                
        # Main path
        # x = torch.exp(x)
        x = F.relu(x)
        x = self.maxpool0(x)
        
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        # x = F.relu(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Rejoin paths
        contextual_weight = self.contextual_weight(x)
        contextual_weight *= is_non_zero_sequence  # This zeroes out contextual weights for all-zero sequences

        combined = contextual_weight * mean_activation
        combined_sum = combined.sum(dim=1, keepdim=True)  # Sum across motifs
        output = self.bias_layer(combined_sum)

        return output, contextual_weight, combined, self.bias_layer.weight

def register_hooks(model):
    activations = {}
    for i, branch in enumerate(model.branches):
        branch.conv0.register_forward_hook(create_hook(activations, f"branch_{i}_conv0"))
        branch.contextual_weight.register_forward_hook(create_hook(activations, f"branch_{i}_contextual_weight"))
    return activations

def create_hook(activations, name):
    """Creates a forward hook function to capture layer outputs."""
    def hook(model, input, output):
        # Detach the output from the computation graph and store it in the activations dictionary
        activations[name] = output.detach()
    return hook
 
# Save motifs to MEME file
def write_motifs_to_meme_file(motifs, file_name, BSSID):
    os.makedirs(ENHANCER_ROOT + "/cell_type_specific_CNN/" + BSSID + "/", exist_ok=True)
    with open(ENHANCER_ROOT + "/cell_type_specific_CNN/" + BSSID + "/" + file_name, 'w') as f:
        f.write('MEME version 5.1.1\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        for i, motif in enumerate(motifs):
            f.write('MOTIF {0} MEME-{0}\n'.format(i))
            f.write('letter-probability matrix: alength=4 w={0} nsites=1 E=0.0e+00\n'.format(motif.shape[0]))
            for row in motif:
                f.write('\t'.join([str(x) for x in row]) + '\n')
            f.write('\n')



# Extract motifs for each cell line
for bssid in nonDupBSSID:
    print("Processing " + bssid)
    filesX = [bssid + ".one_hot.npy"]
    filesy = [bssid + ".y.npy"]
    DIR = ENHANCER_ROOT + "/cell_type_specific_CNN/" + bssid + "/"

    # Load data
    X_enc_list = []
    for file_name in filesX:
        array = np.load(os.path.join(saveXdir, file_name))
        array = array[random_genes]
        X_enc_list.append(array)
    X_enc = np.stack(X_enc_list, axis=0)

    y_enc_list = []
    for file_name in filesy:
        array = np.load(os.path.join(saveydir, file_name))
        array = array[random_genes]
        y_enc_list.append(array)
    y_enc = np.stack(y_enc_list, axis=0)

    # Reshape by merging the first two dimensions and keeping the rest as they are
    X_gene = X_enc.reshape(X_enc.shape[0] * X_enc.shape[1], *X_enc.shape[2:])
    y_gene = y_enc.flatten()
    y_gene_log10 = y_gene / np.log2(10)

    # Go over all 10 models (iterations)
    for iteration in range(10):
        full_model_name = bssid + "_it_" + str(iteration) + "_" + REMARK
        # print("Loading model " + full_model_name)

        # an empty list to store motifs
        pro_motifs = np.empty((0, 8, 4))
        enh_motifs = np.empty((0, 8, 4))
        # Store motif weights
        pro_motif_weights = []
        enh_motif_weights = []

        #process each branch separately due to memory issues
        for br in range(NUM_BRANCHES):  # For each branch
            # print("Branch " + str(br))
            # Recreate the model structure
            X_tensor = torch.tensor(X_gene).float()
            model = MultiBranchModel(num_branches=NUM_BRANCHES)
            model_file = DIR + full_model_name + ".pth"
            # Load the saved state dictionary into the model
            if CUDA:
                model.load_state_dict(torch.load(model_file))
            else:
                model_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
                model.load_state_dict(model_state_dict)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            branch = model.branches[br]
            # print(branch.bias_layer.weight)

            # Register hooks to capture activations. This should be done before the forward pass.
            activations = {}
            branch.conv0.register_forward_hook(create_hook(activations, (br, 'conv0')))
            # branch.combined.register_forward_hook(create_hook(activations, (br, 'combined')))

            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Turn off gradients for prediction
                # Perform a forward pass. Hooks will capture activations during this process.
                output, weight_out, combined_out, bias_out = model(X_tensor.to(device))
            
            # Get contexual weight, total combined weight and motif activations
            # pred_expression = output.cpu().flatten().numpy()
            # print(pred_expression)

            gradeient_out = bias_out.cpu().numpy()
            Contexual_weight_out = weight_out[:, br, :].cpu().numpy()
            Combined_weight_out = combined_out[:, br, :].cpu().numpy()
            Conv1D_1_out = activations[(br, 'conv0')].cpu().numpy()

            # # gene_correlations_pearson = []
            # gene_correlations_spearman = []
            # # Calculate correlation for each gene (across motifs)
            # for gene_idx in range(Contexual_weight_out.shape[0]):
            #     contextual_gene = abs(Contexual_weight_out[gene_idx, :])
            #     combined_gene = abs(Combined_weight_out[gene_idx, :])
                
            #     # Only calculate correlation if both vectors have non-zero variance
            #     if np.var(contextual_gene) > 0 and np.var(combined_gene) > 0:
            #         # pearson_corr, _ = pearsonr(contextual_gene, combined_gene)
            #         spearman_corr, _ = spearmanr(contextual_gene, combined_gene)
                    
            #         # gene_correlations_pearson.append(pearson_corr)
            #         gene_correlations_spearman.append(spearman_corr)

            # # gene_correlations_pearson = np.array(gene_correlations_pearson)
            # gene_correlations_spearman = np.array(gene_correlations_spearman)
            # print(gene_correlations_spearman[CTCF])

            # # Print statistics
            # print(f"Branch {br} - Gene-level correlations:")
            # # print(f"  Pearson: mean={np.mean(gene_correlations_pearson):.4f}, std={np.std(gene_correlations_pearson):.4f}")
            # print(f"  Spearman: mean={np.mean(gene_correlations_spearman):.4f}, std={np.std(gene_correlations_spearman):.4f}")
            # print(f"  Number of genes with valid correlations: {len(gene_correlations_spearman)}")

            # plt.figure(figsize=(8, 6))
            # plt.hist(gene_correlations_spearman, bins=30, alpha=0.7, color='green', edgecolor='black')
            # plt.xlabel('Spearman Correlation')
            # plt.ylabel('Number of Genes')
            # plt.title(f'Branch {br} - Spearman Correlation Distribution\nMean: {np.mean(gene_correlations_spearman):.3f}')
            # plt.axvline(np.mean(gene_correlations_spearman), color='red', linestyle='--', 
            #         label=f'Mean: {np.mean(gene_correlations_spearman):.3f}')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"./multi_agent/{bssid}_branch_{br}_spearman_correlations.png", dpi=300, bbox_inches="tight")
            # plt.close()


            del output, model, X_tensor, 
            # If using CUDA, clear the cache as well
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate the median importance for each motif across all genes
            weight_thres = 0.5 * np.max(Combined_weight_out, axis=0)
            median_weights = np.zeros(Combined_weight_out.shape[1])
            # Iterate through each column and count the values exceeding 50% of the maximum value
            for mot in range(Combined_weight_out.shape[1]):
                # if weight_thres is 0 or smaller, set median weight to 0
                if weight_thres[mot] <= 0:
                    median_weights[mot] = 0
                    continue

                # Create a mask depicting if motif in each gene's importance is greater than threshold
                above_thres_mask = Combined_weight_out[:, mot] > weight_thres[mot]
                median_weights[mot] = np.median(Combined_weight_out[above_thres_mask, mot])

            # Find genes that have a promoter/enhancer for this branch
            non_zero_genes = np.any(Combined_weight_out != 0, axis=1)
            # Check if there are any non-zero rows, if not, skip to the next branch
            if not np.any(non_zero_genes):
                print("No genes have regulatory elements in this branch, continuing to next branch.")
                continue

            Conv1D_1_out_tp = np.transpose(Conv1D_1_out, (1,0,2)) 
            # sorted_indices = np.argsort(-median_weights)

            # Make PWMs
            for mot in range(NUM_MOTIFS):        
                Conv1D_1_out_tp_flat = abs(Conv1D_1_out_tp[mot]).flatten()
                idxLargestActivations = np.argpartition(Conv1D_1_out_tp_flat, -NUM_ACTIVATIONS)[-NUM_ACTIVATIONS:]
                idxRegion, idxStartPos = np.unravel_index(idxLargestActivations, Conv1D_1_out_tp[mot].shape)

                # Calculate the start indices for the subsequences
                start_indices = idxStartPos[:, np.newaxis] + np.arange(MOTIF_LENGTH)
                # Use broadcasting to create an extended idxRegion array
                extended_idxRegion = idxRegion[:, np.newaxis, np.newaxis]
                # Extract subsequences using advanced NumPy indexing for branch 1
                subsequences = X_gene[extended_idxRegion, br, start_indices[:, :, np.newaxis],:]
                # Compute the PWM
                pwm = np.squeeze(np.sum(subsequences, axis=0), axis=1)/len(idxRegion)
                
                # Check if PWM does not add up to 1 (0.98 is the threshold)
                row_sums = np.sum(pwm, axis=1)
                if np.any(row_sums < 0.99):
                    print("Not all rows sum up to 1.")
                    print(mot)
                else:
                    # Find cases with row sum 0.99999 and fix
                    if not np.allclose(row_sums, 1, atol=1e-6):
                        rows_to_fix = ~np.isclose(row_sums, 1, atol=1e-6)
                        # Fix rows by padding the last column
                        pwm[rows_to_fix, -1] += (1 - row_sums[rows_to_fix])
                    # Add to discovered motifs
                    if br < 10:
                        pro_motifs = np.append(pro_motifs, [pwm], axis=0)
                        pro_motif_weights.append(median_weights[mot])
                    else:
                        enh_motifs = np.append(enh_motifs, [pwm], axis=0)
                        enh_motif_weights.append(median_weights[mot])

        #     print("Branch " + str(br) + " done")
        # print("model" + full_model_name + " done")

        # combine similar motifs
        write_motifs_to_meme_file(pro_motifs, full_model_name + '_' + str(iteration) + '_pro_motifs_all_256.meme', bssid)
        np.save(ENHANCER_ROOT + "/cell_type_specific_CNN/" + bssid + "/" + full_model_name + '_' + str(iteration) + '_pro_motifs_weights_all_256.npy', pro_motif_weights)

        # write_motifs_to_meme_file(enh_motifs, full_model_name + '_' + str(iteration) +'_enh_motifs_all_256.meme', bssid)
        # np.save("$ENHANCER_ROOT/cell_type_specific_CNN/" + bssid + "/" + full_model_name + '_' + str(iteration) + '_enh_motifs_weights_all_256.npy', enh_motif_weights)
        # print("done")


    # # Record contextual weight distribution
    # contex_weight_all_pro = np.vstack(contex_weight_all_pro)
    # contex_weight_all_pro = np.median(contex_weight_all_pro, axis=0)
    # contex_weight_all_enh = np.vstack(contex_weight_all_enh)
    # contex_weight_all_enh = np.median(contex_weight_all_enh, axis=0)
    # np.save(f"./cell_type_specific_CNN/{BSSID}/{BSSID}_contex_weight_256all_pro.npy", np.array(contex_weight_all_pro))
    # np.save(f"./cell_type_specific_CNN/{BSSID}/{BSSID}_contex_weight_256all_enh.npy", np.array(contex_weight_all_enh))
 

    # # # contex_weight_all_pro = np.load("./contex_weight_256all_pro.npy")
    # # # contex_weight_all_enh = np.load("./contex_weight_256all_enh.npy")

    # # Create the KDE plot
    # sns.set(style="whitegrid")  # for a nice grid background
    # plt.figure(figsize=(6, 3))  # adjust figure size
    # plt.plot(contex_weight_all_pro, color='royalblue')  # plot smooth curve
    # plt.fill_between(np.arange(NUM_MOTIFS), contex_weight_all_pro, color='royalblue', alpha=0.1)
    # plt.xlabel('Motif')  # set x-label
    # plt.ylabel('Contextual Weight Sum')  # set y-label
    # plt.savefig(f"./cell_type_specific_CNN/{BSSID}/{BSSID}_contex_weight_256all_pro.pdf", dpi=300, bbox_inches="tight")
    # plt.close()

    # # Create the KDE plot
    # sns.set(style="whitegrid")  # for a nice grid background
    # plt.figure(figsize=(6, 3))  # adjust figure size
    # plt.plot(contex_weight_all_enh, color='green')  # plot smooth curve
    # plt.fill_between(np.arange(NUM_MOTIFS), contex_weight_all_enh, color='green', alpha=0.1)
    # plt.xlabel('Motif')  # set x-label
    # plt.ylabel('Contextual Weight Sum')  # set y-label
    # plt.savefig(f"./cell_type_specific_CNN/{BSSID}/{BSSID}_contex_weight_256all_enh.pdf", dpi=300, bbox_inches="tight")
    # plt.close()



