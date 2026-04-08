import os

infile = f'coffea/test_pu0_barrel_genPtCut.coffea'
stage = f'cut6_splitMatched'
outdir = f'debug_plots_tt200'

#cmd = f'python plot_cone_overlay_matched.py {infile} --stage {stage} --outdir {outdir} --which both --density'

cmd = f'python plot_cone_overlay_matchedonly.py {infile} \
  --stage {stage} \
  --outdir {outdir} \
  --which both \
  --density \
  --linear'


print(cmd)
os.system(cmd)
