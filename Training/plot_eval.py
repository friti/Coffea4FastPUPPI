import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
args = parser.parse_args()

# Plot loss
path = os.path.join(args.run_dir, "loss_history.npz")
data = np.load(path)

epoch = data["epoch"]
train = data["train_loss"]
val_mse = data["val_loss_mse"]
val_huber = data["val_loss_huber"]

plt.figure()
plt.plot(epoch, train, label="train_huber")
plt.plot(epoch, val_huber, label="val_huber")
#plt.plot(epoch, val_mse, label="val_mse")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

out = os.path.join(args.run_dir, "loss_plot.png")
plt.savefig(out)
print(f"Saved: {out}")

# Plot the 2D: target vs prediction
files = glob.glob(f'{args.run_dir}/outputs_epoch*.npz')

for f in files:
    data = np.load(f)

    pred = data["pred"]
    target = data["target"]

    plt.figure()

    h = plt.hist2d(target, pred, bins=2000, norm='log')
    plt.colorbar()


    #plt.scatter(target, pred, s=2, alpha=0.3)

    #mn = min(target.min(), pred.min())
    #mx = max(target.max(), pred.max())
    mn = 0
    mx = 2

    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlim([mn, mx])
    plt.ylim([mn, mx])

    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Pred vs Target")
    plt.grid()

    out = f.replace(".npz", "_scatter.png")
    plt.savefig(out)
    print(f"Saved: {out}")

    res = pred - target

    plt.figure()
    plt.hist(res, bins=1000)
    plt.xlabel("Prediction - Target")
    plt.ylabel("n(Neutrals)")
    plt.title("Residuals")

    plt.xlim([-1,1])
    out = f.replace(".npz", "_residuals.png")
    plt.savefig(out)
    print(f"Saved: {out}")
