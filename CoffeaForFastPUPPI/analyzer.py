import os, glob, yaml, argparse, importlib
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import coffea.util as util

from processor import P2L1TAnalyzer
from histo_config.expand_histo import expand_histo_yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rootDir", required=True, help="Directory OR single .root file")
    parser.add_argument("--filePattern", default="*.root", help="Used if rootDir is a directory")
    parser.add_argument("--treename", default="Events")

    parser.add_argument("--outName", required=True)
    parser.add_argument("--histoConfig", required=True)

    parser.add_argument("--cutConfig", required=True,
                        help="Python module path, e.g. cut_config.cut_config")

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maxFile", type=int, default=-1)

    args = parser.parse_args()

    # hist yaml
    with open(args.histoConfig) as f:
        raw = yaml.safe_load(f)
    hist_config = expand_histo_yaml(raw)

    # load cut config module
    cutflow_module = importlib.import_module(args.cutConfig)

    # files
    if args.rootDir.endswith(".root") and os.path.isfile(args.rootDir):
        file_list = [args.rootDir]
    else:
        file_list = glob.glob(os.path.join(args.rootDir, args.filePattern))

    if not file_list:
        raise RuntimeError(f"No ROOT files found: rootDir={args.rootDir}, pattern={args.filePattern}")

    fileset = {"files": {"files": file_list}}
    
    if args.maxFile != -1:    
        fileset = {"files": {"files": file_list[:args.maxFile]}}

    print(f"Input: {args.rootDir} ({len(file_list)} files), treename={args.treename}")
    print(f"Stages: {[name for name, _ in cutflow_module.CUTFLOW]}")

    proc = P2L1TAnalyzer(hist_config=hist_config, cut_config_module=cutflow_module)

    exe = processor.FuturesExecutor(workers=args.workers, merging=True)
    runner = processor.Runner(executor=exe, schema=NanoAODSchema)

    result = runner(fileset=fileset, treename=args.treename, processor_instance=proc)
    
    # ---- build plot_cfg sidecar  ----
    plot_cfg = {}
    hists_by_stage = result.get("hists", {}) or {}

    for stage, hdict in hists_by_stage.items():
        plot_cfg[stage] = {}
        for hname in hdict.keys():
            cfg = hist_config.get(hname, {}) or {}
            plot_cfg[stage][hname] = {"logy": bool(cfg.get("logy", False))}

    result["plot_cfg"] = plot_cfg

    os.makedirs("coffea", exist_ok=True)
    outpath = f"coffea/{args.outName}.coffea"
    util.save(result, outpath)
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
