"""
cem_train_v7.py  —  Resumable, dependency-free trainer (Cross-Entropy Method).

Trains a small MLP policy (obs -> 32 tanh -> 6 tanh actions) on the ClimbBot env without
torch/stable-baselines3. State is checkpointed to disk so you can train in chunks: each
run loads the previous checkpoint, trains for a wall-clock budget, and saves progress,
the best policy weights, a learning curve, and a CSV history.

Run (repeat to keep training):
    python cem_train_v7.py --tag stage0 --seconds 35
    python cem_train_v7.py --tag stage0 --seconds 35   # resumes
Play a saved policy in the viewer (on a machine with a display):
    python cem_train_v7.py --tag stage0 --play
"""
import os, time, csv, argparse, numpy as np
from climb_env_v7 import ClimbBotEnv

H = 32  # hidden units


def shapes(odim, adim):
    return [(H, odim), (H,), (adim, H), (adim,)]


def unpack(p, odim, adim):
    s = shapes(odim, adim); i = 0; out = []
    for sh in s:
        n = int(np.prod(sh)); out.append(p[i:i + n].reshape(sh)); i += n
    return out  # W1,b1,W2,b2


def policy(p, obs, odim, adim):
    W1, b1, W2, b2 = unpack(p, odim, adim)
    h = np.tanh(W1 @ obs + b1)
    return np.tanh(W2 @ h + b2)


def rollout(env, p, odim, adim, seed, ep_len):
    obs, _ = env.reset(seed=seed)
    ret = 0.0; dmin_r = dmin_l = 9.0; eng = 0; succ = 0
    for _ in range(ep_len):
        obs, r, term, trunc, info = env.step(policy(p, obs, odim, adim))
        ret += r
        dmin_r = min(dmin_r, info["d_r"]); dmin_l = min(dmin_l, info["d_l"])
        eng += int(info["engaged_r"] and info["engaged_l"])
        succ += int(info["is_success"])
        if term or trunc:
            break
    return ret, dmin_r, dmin_l, eng / ep_len, succ / ep_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage0")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--pop", type=int, default=28)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--ep-len", type=int, default=120)
    ap.add_argument("--evals", type=int, default=1, help="episodes averaged per candidate (less noise)")
    ap.add_argument("--climb", action="store_true", help="free-base climbing from a hang start")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--init-qpos", default="hang_start_qpos.npy")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--outdir", default="runs_v7")
    args = ap.parse_args()

    outdir = os.path.join(args.outdir, "cem_" + args.tag)
    os.makedirs(outdir, exist_ok=True)
    ckpt = os.path.join(outdir, "state.npz")
    freeze = (not args.no_freeze) and (not args.climb)

    def make(max_steps, render_mode=None):
        kw = dict(freeze_base=freeze, max_steps=max_steps, init_joint_noise=0.03, render_mode=render_mode)
        if args.xml: kw["xml_path"] = args.xml
        if args.climb:
            kw.update(climb=True, init_qpos_file=args.init_qpos, init_joint_noise=0.02)
        return ClimbBotEnv(**kw)

    env = make(args.ep_len)
    odim = env.observation_space.shape[0]; adim = env.action_space.shape[0]
    dim = sum(int(np.prod(s)) for s in shapes(odim, adim))

    if args.play:
        d = np.load(os.path.join(outdir, "best_policy.npz"))
        p = d["p"]; penv = make(2000, render_mode="human")
        obs, _ = penv.reset()
        for _ in range(4000):
            obs, r, term, trunc, info = penv.step(policy(p, obs, odim, adim))
            if term or trunc: obs, _ = penv.reset()
        return

    if os.path.exists(ckpt):
        st = np.load(ckpt, allow_pickle=True)
        mu = st["mu"]; sigma = st["sigma"]; best_p = st["best_p"]; best_ret = float(st["best_ret"])
        hist = st["hist"].tolist(); it0 = int(st["it"])
        print(f"resumed {args.tag} at iter {it0}, best_ret={best_ret:.1f}")
    else:
        mu = np.zeros(dim); sigma = np.full(dim, 0.4); best_p = mu.copy()
        best_ret = -1e9; hist = []; it0 = 0
        print(f"fresh start {args.tag} (dim={dim})")

    rng = np.random.default_rng(1234 + it0)
    t0 = time.time(); it = it0
    while time.time() - t0 < args.seconds:
        pop = rng.normal(mu, sigma, size=(args.pop, dim))
        res = []
        for j in range(args.pop):
            evs = [rollout(env, pop[j], odim, adim, seed=it * 1000 + j * 7 + e, ep_len=args.ep_len)
                   for e in range(args.evals)]
            avg = tuple(np.mean([e[k] for e in evs]) for k in range(5))
            res.append((avg, j))
        res.sort(key=lambda x: -x[0][0])
        elite = pop[[r[1] for r in res[:args.elite]]]
        mu = elite.mean(0); sigma = elite.std(0) + 0.02
        if res[0][0][0] > best_ret:
            best_ret = res[0][0][0]; best_p = pop[res[0][1]].copy()
        mret = np.mean([r[0][0] for r in res])
        mdr = np.mean([r[0][1] for r in res]); mdl = np.mean([r[0][2] for r in res])
        meng = np.mean([r[0][3] for r in res]); msuc = np.mean([r[0][4] for r in res])
        hist.append([it, mret, res[0][0][0], mdr, mdl, meng, msuc])
        print(f"iter {it:3d} | mean {mret:8.1f} | best {res[0][0][0]:8.1f} | "
              f"d_r {mdr:.3f} d_l {mdl:.3f} | eng {meng*100:3.0f}% | succ {msuc*100:3.0f}%")
        it += 1

    np.savez(ckpt, mu=mu, sigma=sigma, best_p=best_p, best_ret=best_ret,
             hist=np.array(hist, dtype=object), it=it)
    np.savez(os.path.join(outdir, "best_policy.npz"), p=best_p, odim=odim, adim=adim, H=H)
    with open(os.path.join(outdir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter", "mean_ret", "best_ret", "d_r", "d_l", "eng", "succ"])
        w.writerows(hist)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = np.array(hist, dtype=float)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(h[:, 0], h[:, 1], "-o", label="mean return")
        ax[0].plot(h[:, 0], h[:, 2], "-s", label="best return")
        ax[0].set_xlabel("iter"); ax[0].set_ylabel("return"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[0].set_title(f"CEM {args.tag} — return")
        ax[1].plot(h[:, 0], h[:, 3], label="d_r (m)"); ax[1].plot(h[:, 0], h[:, 4], label="d_l (m)")
        ax[1].axhline(0.06, ls="--", c="g")
        ax2 = ax[1].twinx(); ax2.plot(h[:, 0], h[:, 6] * 100, "-^", c="purple"); ax2.set_ylabel("success %", color="purple")
        ax[1].set_xlabel("iter"); ax[1].set_ylabel("dist (m)"); ax[1].grid(alpha=.3); ax[1].legend(loc="upper right")
        ax[1].set_title("reach + success")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "curve.png"), dpi=120)
        print("saved", os.path.join(outdir, "curve.png"))
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nDONE through iter {it-1} | best_ret={best_ret:.1f} | saved to {outdir}")


if __name__ == "__main__":
    main()
