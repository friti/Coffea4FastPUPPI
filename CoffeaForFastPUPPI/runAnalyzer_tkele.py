import os

histo_config = "histo_config/histo_tkeleLeadSub.yaml"

cutConfig = "cut_config.cut_config_tkele"

mass_points = [5, 10, 15, 20, 30]

for mass in mass_points:
    rootDir = f'/eos/cms/store/cmst3/group/l1tr/kypark/Xee_isolationStudies/HAHM_ZdToEE_m{mass}_pu200//FP/151X/'
    outName = f"tkele_{mass}_barrel_newIsoDzOtherEleVeto_pt4_finerBin_nBin50k_2D"

    cmd = (
        f"python analyzer.py "
        f"--rootDir {rootDir} "
        f"--histoConfig {histo_config} "
        f"--cutConfig {cutConfig} "
        f"--outName {outName} "
        #f"--maxFile 500 "
        )

    print(cmd)
    os.system(cmd)

