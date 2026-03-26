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

# MET computation

- The notebook `Reproduce_MET.ipynb` includes code to check that the MET and PUPPI computations on top of fastPUPPI ntuples is correct and it works.
- The notebook `MET_pipeline.ipynb` runs on fastPUPPI ntuples and computes the new PUPPi collection with new targets, compares it with the original PUPPI collection, and computes the MET for both.

# Jet validation

- The notebook `reclusteringSCJets.ipynb` includes code to perform a reclustering to produce seeded-cone jets starting from two different collections
