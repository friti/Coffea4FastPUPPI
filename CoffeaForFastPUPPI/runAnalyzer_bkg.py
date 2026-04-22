import os

histo_config = "histo_config/histo_bkg_tkele.yaml"
cut_config = "cut_config.cut_config_bkg_tkele"

#root_dir = "/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples/NuGunAllEta_PU200//FP/151X_TkElePtRegrTemp_A1/"
root_dir = "/eos/cms/store/cmst3/group/l1tr/kypark/Xee_isolationStudies/NuGunAllEta_PU200/FP/151X/"

outName = "bkg_tkele_barrel_newIsoDzOtherEleVeto_pt4_finerBin_nBin50k_2D"

cmd = (
    f"python analyzer.py "
    f"--rootDir {root_dir} "
    f"--histoConfig {histo_config} "
    f"--cutConfig {cut_config} "
    f"--outName {outName} "
    f"--maxFile 200 "
)

print(cmd)
os.system(cmd)
