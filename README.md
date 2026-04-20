# Recipe for FastPUPPI framework

```
cmsrel CMSSW_15_1_0_pre4
cd CMSSW_15_1_0_pre4/src
cmsenv
git cms-init
git cms-addpkg DataFormats/L1TParticleFlow
git cms-addpkg DataFormats/L1TCorrelator
git cms-addpkg L1Trigger/Phase2L1ParticleFlow
git cms-addpkg L1Trigger/TrackTrigger
git cms-addpkg SimTracker/TrackTriggerAssociation
git cms-addpkg L1Trigger/VertexFinder

git cms-checkout-topic -u friti:new_G_branches_and_nnvtx_on

git clone git@github.com:elfontan/FastPUPPI.git -b puppiML_target
```

It includes 
-  puppi weight and NNVtx score in the PUPPI collection
-  NNVtx association ON for the charged tracks
-  PUPPI collection is the same as PF candidates (useful for PUPPI studies)
-  New sums and ratios are saved in the ntuples for a dR of 0.2

# Notebooks
## MET computation

- The notebook `Reproduce_MET.ipynb` includes code to check that the MET and PUPPI computations on top of fastPUPPI ntuples is correct and it works.
- The notebook `MET_pipeline.ipynb` runs on fastPUPPI ntuples and computes the new PUPPi collection with new targets, compares it with the original PUPPI collection, and computes the MET for both.

## Jet validation

- The notebook `reclusteringSCJets.ipynb` includes code to perform a reclustering to produce seeded-cone jets starting from two different collections

# CoffeaForFastPUPPI
## Run coffea on the ntuples
Last update: 2026 April 7th
- The coffea framework will read the FastPUPPI ntuples and make cuts & plot stuff.
- Plan is to update cut & histogramming configs to streamline a bit better, but here's a working version.

### Workflow

1. Open `runAnalyzer_pf.py` (or for background, `runAnalyzer_bkg.py`). 
- Add the input file location or the input file itself. For example, you can put the location of the perfNano.root files or the hadded root file itself. 
- Update the output coffea file name.
- Choose `histo_config` and `cut_config`

2. For `histo_config`: it's a bit messy but choose the `yaml` file and update it per needed.

3. For `cut_config`: most of functions needed for applying relevant cuts are implemented in `cut_config_pf.py` (KP: let me know if more features are needed). Change the bottom of the file `CUTFLOW` to modify any cuts to be applied. Top of the file lists some cut values.

4. Run `python3 runAnalyzer_pf.py`. 

5. Plotting & cutflow related scripts can be found in `plotting` repo. (TO-DO: Add example of these codes)


# Training
## (Preliminary) GNN training for regression
Preliminary training on GNN (EdgeConv, GravNet) can be found in the `Training` repo.

1. Set up environment on lxplus:
    ```
    ssh <your_username>@lxplus-gpu.cern.ch

    apptainer shell -B /afs -B /eos -B /eos/user/k/kypark -B /eos/home-k/kypark -B /afs/cern.ch/user/k/kypark -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest --bind /usr/local/cuda-12.4:/usr/local/cuda-12.4

    python -m venv --system-site-packages bestpuppi
    source bestpuppi/bin/activate
    export CUDA_HOME=/usr/local/cuda-12.4

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

    ```

2. Prepare dataset
    ```
    # Example
    python3 prepare_dataset_batches.py --input perfNano_TTbar_PU200.root --output_dir ./graphs --graphs_per_file 20000
    ```
    - To add more inputs, change this script (once the branches are available on the ntuples)
    - Some pre-processed inputs (1000 neutral cones per pt file) can be found here: `/eos/cms/store/group/cmst3/group/l1tr/kypark/puppi_training_neutral_dr0p3/`

3. Run training
    ```
    # Example
    python3 train_gravnet.py --data "./graphs/*.pt" --epochs 20 --batch_size 128

    ```

    It will run the training and save at every epoch the training output & model at `runs/runs/run_*_*` directory. 

4. Check training: plot loss history, regression vs. target
    ```
    python3 plot_eval.py --run runs/runs/run_*_* 
    ```
