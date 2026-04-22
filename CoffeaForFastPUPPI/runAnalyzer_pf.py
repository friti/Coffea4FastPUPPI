import os

histo_config = "histo_config/histo_pfpuppi.yaml"
cut_config = "cut_config.cut_config_pf"

root_dir = "/eos/cms/store/cmst3/group/l1tr/elfontan/l1tPFplusPuppi/fp_ntuples_NNVtx_151X/TT_PU0/FP/151Xv0/" # PU 0

outName = "pu0_barrel_genPtCut"

cmd = (
    f"python analyzer.py "
    f"--rootDir {root_dir} "
    f"--histoConfig {histo_config} "
    f"--cutConfig {cut_config} "
    f"--outName {outName} "
)

print(cmd)
os.system(cmd)
