import json
import torch
import numpy as np
from datetime import datetime
from collections import namedtuple

def timer(func):
    def wrapper(*args, **kwds):
        start_t = datetime.now()
        rets = func(*args, **kwds)
        end_t = datetime.now()
        if rets is not None:
            return (*rets, end_t-start_t)
        else:
            return end_t-start_t
    return wrapper

def build_config(model_path, replica):
    config_file = model_path / replica / 'config.json'
    stats_file = model_path / 'stats_train_s35.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    with open(stats_file, 'r') as f:
        norm_stats = json.load(f)

    default = {
        'torsion_multiplier': 0, 'collapsed_batch_norm': False,
        'filters_1d': [], 'is_ca_feature': False,
    }
    config['norm_stats'] = norm_stats
    config['network_config'] = {**default, **config['network_config']}
    exclude = norm_stats.keys()

    def make_nt(d, n):
        return namedtuple(n, d.keys())(**{k: make_nt(v, k) if isinstance(v, dict) and k not in exclude else v for k,v in d.items()})

    return make_nt(config, 'config')

@timer
def load_tf_ckpt(model, model_file_pt):
    import tensorflow as tf

    model_file_tf = model_file_pt.parent / 'tf_graph_data' / 'tf_graph_data.ckpt'
    for tf_name, tf_shape in tf.train.list_variables(str(model_file_tf)):
        tf_var = tf.train.load_variable(str(model_file_tf), tf_name)
        
        main_module, *others = tf_name.split('/')
        if main_module == 'ca_cb_logits': continue
        elif main_module.startswith('collapsed_embed'):
            n = int(main_module.split('_')[-1])
            pointer = model.collapsed_embed[n][1 if others[0] == 'BatchNorm' else 0]
        else:
            pointer = getattr(model, main_module)
            if main_module in ['Deep2D', 'Deep2DExtra']:
                if others[0].startswith('conv'):
                    pointer = getattr(pointer, others[0]).conv
                elif others[0].startswith('res'):
                    pointer = getattr(pointer, others[0].split('_')[0])
                    if others[0].endswith('1x1'):
                        pointer = pointer.conv_1x1
                    elif others[0].endswith('1x1h'):
                        pointer = pointer.conv_1x1h
                    elif others[0].endswith('3x3h'):
                        pointer = pointer.conv_3x3h

                    if others[-1] in ['w', 'b']:
                        pointer = pointer.conv
                    else:
                        pointer = pointer.bn
                elif others[0] == 'output_reshape_1x1h':
                    pointer = model.output_reshape_1x1h
                    if len(others) > 2:
                        pointer = pointer.bn
                    else:
                        pointer = pointer.conv

        if others:
            if others[-1] in ['weights', 'w']:
                pointer = pointer.weight
                if len(tf_var.shape) == 2:
                    # linear w
                    tf_var = tf_var.transpose()
                elif len(tf_var.shape) == 4:
                    #    tf conv w: [filter_height, filter_width, in_channels, out_channels]
                    # torch conv w: [out_channels, in_channels, filter_height, filter_width]
                    tf_var = tf_var.transpose((3, 2, 0, 1))
            elif others[-1] in ['biases', 'b']:
                pointer = pointer.bias
            elif others[-1] == 'beta':
                pointer = pointer.bias
            elif others[-1] == 'moving_mean':
                pointer = pointer.running_mean
            elif others[-1] == 'moving_variance':
                pointer = pointer.running_var

        try:
            assert pointer.shape == tf_var.shape
        except AssertionError as e:
            print(main_module, others)
            e.args += (pointer.shape, tf_var.shape)
            raise

        pointer.data = torch.from_numpy(tf_var)

    # save pytorch model
    torch.save(model.state_dict(), model_file_pt)

def save_seq_prob(prob, seq, out_file):
    SECONDARY_STRUCTURES = '-HETSGBI'
    if len(prob.shape) == 1:
        prob = prob.reshape(-1,1)
    L, n = prob.shape
    label = 'asa' if n == 1 else 'secstruct'

    with out_file.open('w') as f:
        f.write(f"# LABEL {label} CLASSES [{''.join(SECONDARY_STRUCTURES[:n])}]\n\n")
        for i in range(L):
            ss = SECONDARY_STRUCTURES[prob[i].argmax()]
            f.write(f"{i+1:4d} {seq[i]:1s} {ss:1s} {''.join(['%6.3f'%p for p in prob[i]])}\n")

def generate_domains(target, seq, crop_sizes='64,128,256', crop_step=32):
    windows = [int(x) for x in crop_sizes.split(",")]
    num_residues = len(seq)
    domains = []
    domains.append({"name": target, "description": (1, num_residues)})

    for window in windows:
        starts = list(range(0, num_residues - window, crop_step))
        if num_residues >= window:
            starts += [num_residues - window]
        for start in starts:
            name = f'{target}-l{window}_s{start}'
            domains.append({"name": name, "description": (start + 1, start + window)})
    
    return domains

def save_rr_file(probs, seq, domain, filename):
    assert len(seq) == probs.shape[0]
    assert len(seq) == probs.shape[1]
    
    with open(filename, 'w') as f:
        f.write(f'PFRMAT RR\nTARGET {domain}\nAUTHOR DM-ORIGAMI-TEAM\nMETHOD Alphafold - PyTorch\nMODEL 1\n{seq}\n')
        for i in range(probs.shape[0]):
            for j in range(i + 1, probs.shape[1]):
                f.write(f'{i+1:d} {j+1:d} 0 8 {probs[j,i]:f}\n')
        f.write('END\n')

def plot_contact_map(target, mats, out):
    import matplotlib.pyplot as plt
    if len(mats) == 1:
        fig, ax = plt.subplots()
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, len(mats), figsize=(11*len(mats),8))
    fig.subplots_adjust(wspace=0)

    for i, mat in enumerate(mats):
        if len(mat.shape) == 3 and mat.shape[-1] == 64:
            vmax = mat.shape[-1] - 1
            mat = mat.argmax(-1)
            im = axs[i].imshow(mat, cmap=plt.cm.Blues_r, vmin=0, vmax=vmax)
            cb = fig.colorbar(im, ax=axs[i])
            cb.set_ticks(np.linspace(0, vmax, 11))
            cb.set_ticklabels(range(2, 23, 2))
            if len(mats) != 1:
                axs[i].set_title('distance', fontsize=20)
        else:
            im = axs[i].imshow(mat, cmap=plt.cm.Blues, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=axs[i])
            if len(mats) != 1:
                axs[i].set_title('contact', fontsize=20)

    if len(mats) == 1:
        plt.title(target)
        plt.savefig(out, dpi=300)
    else:
        fig.suptitle(target, fontsize=20)
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.5)
        #@title Run AlphaFold and download prediction

#@markdown Once this cell has been executed, a zip-archive with 
#@markdown the obtained prediction will be automatically downloaded 
#@markdown to your computer.

# --- Run the model ---
model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_2_ptm']

def _placeholder_template_feats(num_templates_, num_res_):
  return {
      'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
      'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
      'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37], np.float32),
      'template_domain_names': np.zeros([num_templates_], np.float32),
      'template_sum_probs': np.zeros([num_templates_], np.float32),
  }

output_dir = 'prediction'
os.makedirs(output_dir, exist_ok=True)

plddts = {}
pae_outputs = {}
unrelaxed_proteins = {}

with tqdm.notebook.tqdm(total=len(model_names) + 1, bar_format=TQDM_BAR_FORMAT) as pbar:
  for model_name in model_names:
    pbar.set_description(f'Running {model_name}')
    num_templates = 0
    num_res = len(sequence)

    feature_dict = {}
    feature_dict.update(pipeline.make_sequence_features(sequence, 'test', num_res))
    feature_dict.update(pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))
    feature_dict.update(_placeholder_template_feats(num_templates, num_res))

    cfg = config.model_config(model_name)
    params = data.get_model_haiku_params(model_name, './alphafold/data')
    model_runner = model.RunModel(cfg, params)
    processed_feature_dict = model_runner.process_features(feature_dict,
                                                           random_seed=0)
    prediction_result = model_runner.predict(processed_feature_dict)

    mean_plddt = prediction_result['plddt'].mean()

    if 'predicted_aligned_error' in prediction_result:
      pae_outputs[model_name] = (
          prediction_result['predicted_aligned_error'],
          prediction_result['max_predicted_aligned_error']
      )
    else:
      # Get the pLDDT confidence metrics. Do not put pTM models here as they
      # should never get selected.
      plddts[model_name] = prediction_result['plddt']

    # Set the b-factors to the per-residue plddt.
    final_atom_mask = prediction_result['structure_module']['final_atom_mask']
    b_factors = prediction_result['plddt'][:, None] * final_atom_mask
    unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                prediction_result,
                                                b_factors=b_factors)
    unrelaxed_proteins[model_name] = unrelaxed_protein

    # Delete unused outputs to save memory.
    del model_runner
    del params
    del prediction_result
    pbar.update(n=1)

  # --- AMBER relax the best model ---
  pbar.set_description(f'AMBER relaxation')
  amber_relaxer = relax.AmberRelaxation(
      max_iterations=0,
      tolerance=2.39,
      stiffness=10.0,
      exclude_residues=[],
      max_outer_iterations=20)
  # Find the best model according to the mean pLDDT.
  best_model_name = max(plddts.keys(), key=lambda x: plddts[x].mean())
  relaxed_pdb, _, _ = amber_relaxer.process(
      prot=unrelaxed_proteins[best_model_name])
  pbar.update(n=1)  # Finished AMBER relax.

# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
banded_b_factors = []
for plddt in plddts[best_model_name]:
  for idx, (min_val, max_val, _) in enumerate(PLDDT_BANDS):
    if plddt >= min_val and plddt <= max_val:
      banded_b_factors.append(idx)
      break
banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
to_visualize_pdb = utils.overwrite_b_factors(relaxed_pdb, banded_b_factors)


# Write out the prediction
pred_output_path = os.path.join(output_dir, 'selected_prediction.pdb')
with open(pred_output_path, 'w') as f:
  f.write(relaxed_pdb)


# --- Visualise the prediction & confidence ---
show_sidechains = True
def plot_plddt_legend():
  """Plots the legend for pLDDT."""
  thresh = [
            'Very low (pLDDT < 50)',
            'Low (70 > pLDDT > 50)',
            'Confident (90 > pLDDT > 70)',
            'Very high (pLDDT > 90)']

  colors = [x[2] for x in PLDDT_BANDS]

  plt.figure(figsize=(2, 2))
  for c in colors:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False, loc='center', fontsize=20)
  plt.xticks([])
  plt.yticks([])
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  plt.title('Model Confidence', fontsize=20, pad=20)
  return plt

# Color the structure by per-residue pLDDT
color_map = {i: bands[2] for i, bands in enumerate(PLDDT_BANDS)}
view = py3Dmol.view(width=800, height=600)
view.addModelsAsFrames(to_visualize_pdb)
style = {'cartoon': {
    'colorscheme': {
        'prop': 'b',
        'map': color_map}
        }}
if show_sidechains:
  style['stick'] = {}
view.setStyle({'model': -1}, style)
view.zoomTo()

grid = GridspecLayout(1, 2)
out = Output()
with out:
  view.show()
grid[0, 0] = out

out = Output()
with out:
  plot_plddt_legend().show()
grid[0, 1] = out

display.display(grid)

# Display pLDDT and predicted aligned error (if output by the model).
if pae_outputs:
  num_plots = 2
else:
  num_plots = 1

plt.figure(figsize=[8 * num_plots, 6])
plt.subplot(1, num_plots, 1)
plt.plot(plddts[best_model_name])
plt.title('Predicted LDDT')
plt.xlabel('Residue')
plt.ylabel('pLDDT')

if num_plots == 2:
  plt.subplot(1, 2, 2)
  pae, max_pae = list(pae_outputs.values())[0]
  plt.imshow(pae, vmin=0., vmax=max_pae, cmap='Greens_r')
  plt.colorbar(fraction=0.046, pad=0.04)
  plt.title('Predicted Aligned Error')
  plt.xlabel('Scored residue')
  plt.ylabel('Aligned residue')

# Save pLDDT and predicted aligned error (if it exists)
pae_output_path = os.path.join(output_dir, 'predicted_aligned_error.json')
if pae_outputs:
  # Save predicted aligned error in the same format as the AF EMBL DB
  rounded_errors = np.round(pae.astype(np.float64), decimals=1)
  indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
  indices_1 = indices[0].flatten().tolist()
  indices_2 = indices[1].flatten().tolist()
  pae_data = json.dumps([{
      'residue1': indices_1,
      'residue2': indices_2,
      'distance': rounded_errors.flatten().tolist(),
      'max_predicted_aligned_error': max_pae.item()
  }],
                        indent=None,
                        separators=(',', ':'))
  with open(pae_output_path, 'w') as f:
    f.write(pae_data)


# --- Download the predictions ---
!zip -q -r {output_dir}.zip {output_dir}
files.download(f'{output_dir}.zip')
