"""
prove_learning_cem.py  —  Dependency-free proof that the redesigned env is learnable.

This does NOT need torch/stable-baselines3. It trains a tiny linear policy with the
Cross-Entropy Method (CEM) on the Stage-0 reach+hook task and shows that episodic
return rises, gripper->hold distance falls, and hook-engagement rate climbs well above
a random baseline. It's a sanity check on the environment + reward shaping itself.

For real training (deeper policy, free-climbing stages) use train_v7.py (PPO).

Run:  python prove_learning_cem.py
"""
import time, numpy as np
from climb_env_v7 import ClimbBotEnv

EP_LEN = 120
N_ITERS = 12
POP = 16
ELITE = 4
SEED = 0


def rollout(env, W, b, seed):
    obs, _ = env.reset(seed=seed)
    ret = 0.0
    dmin_r, dmin_l, eng = 9.0, 9.0, 0
    for _ in range(EP_LEN):
        a = np.tanh(W @ obs + b)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        dmin_r = min(dmin_r, info["d_r"]); dmin_l = min(dmin_l, info["d_l"])
        eng += int(info["engaged_r"] and info["engaged_l"])
        if term or trunc:
            break
    return ret, dmin_r, dmin_l, eng / EP_LEN


def main():
    rng = np.random.default_rng(SEED)
    env = ClimbBotEnv(freeze_base=True, max_steps=EP_LEN, init_joint_noise=0.03)
    odim = env.observation_space.shape[0]; adim = env.action_space.shape[0]
    nW = adim * odim
    dim = nW + adim

    # ---- random-policy baseline ----
    base = []
    for k in range(20):
        p = rng.normal(0, 0.5, dim)
        W = p[:nW].reshape(adim, odim); b = p[nW:]
        base.append(rollout(env, W, b, seed=1000 + k)[0])
    base_mean = np.mean(base)

    mu = np.zeros(dim); sigma = np.full(dim, 0.5)
    history = []
    t0 = time.time()
    best_ret, best_p = -1e9, mu.copy()
    for it in range(N_ITERS):
        pop = rng.normal(mu, sigma, size=(POP, dim))
        results = []
        for j in range(POP):
            W = pop[j, :nW].reshape(adim, odim); b = pop[j, nW:]
            ret, dr, dl, eng = rollout(env, W, b, seed=it * 100 + j)
            results.append((ret, dr, dl, eng, j))
        results.sort(key=lambda x: -x[0])
        elite_idx = [r[4] for r in results[:ELITE]]
        elites = pop[elite_idx]
        mu = elites.mean(0); sigma = elites.std(0) + 0.02
        top = results[0]
        if top[0] > best_ret:
            best_ret = top[0]; best_p = pop[top[4]].copy()
        mean_ret = np.mean([r[0] for r in results])
        mean_dr = np.mean([r[1] for r in results]); mean_dl = np.mean([r[2] for r in results])
        mean_eng = np.mean([r[3] for r in results])
        history.append((it, mean_ret, top[0], mean_dr, mean_dl, mean_eng))
        print(f"iter {it:2d} | mean_ret {mean_ret:8.1f} | best_ret {top[0]:8.1f} "
              f"| d_r {mean_dr:.3f} | d_l {mean_dl:.3f} | engaged {mean_eng*100:4.0f}%")

    dt = time.time() - t0
    # evaluate best deterministically
    W = best_p[:nW].reshape(adim, odim); b = best_p[nW:]
    final = [rollout(env, W, b, seed=5000 + k) for k in range(10)]
    f_ret = np.mean([f[0] for f in final])
    f_dr = np.mean([f[1] for f in final]); f_dl = np.mean([f[2] for f in final])
    f_eng = np.mean([f[3] for f in final])
    env.close()

    print("\n================ RESULT ================")
    print(f"random-policy baseline return : {base_mean:8.1f}")
    print(f"CEM best-policy return (eval)  : {f_ret:8.1f}")
    print(f"improvement factor             : {f_ret/abs(base_mean):.2f}x  (less negative / higher = better)")
    print(f"final min dist  R={f_dr:.3f} m  L={f_dl:.3f} m  (grasp threshold 0.060 m)")
    print(f"final both-hooks-engaged rate  : {f_eng*100:.0f}% of steps")
    print(f"CEM wall-clock                 : {dt:.1f}s for {N_ITERS*POP*EP_LEN} env-steps")

    # ---- save learning curve ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.array(history)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(h[:, 0], h[:, 1], "-o", label="population mean")
        ax[0].plot(h[:, 0], h[:, 2], "-s", label="best of population")
        ax[0].axhline(base_mean, ls="--", c="gray", label="random baseline")
        ax[0].set_xlabel("CEM iteration"); ax[0].set_ylabel("episode return")
        ax[0].set_title("ClimbBot v7 — Stage-0 reach+hook is learnable"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(h[:, 0], h[:, 3], "-o", label="right hand→hold (m)")
        ax[1].plot(h[:, 0], h[:, 4], "-o", label="left hand→hold (m)")
        ax[1].axhline(0.06, ls="--", c="green", label="grasp threshold")
        ax2 = ax[1].twinx(); ax2.plot(h[:, 0], h[:, 5] * 100, "-^", c="purple", label="engaged %")
        ax2.set_ylabel("both hooks engaged (%)", color="purple")
        ax[1].set_xlabel("CEM iteration"); ax[1].set_ylabel("min gripper→hold distance (m)")
        ax[1].set_title("Reaching and engaging the holds"); ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        out = "learning_curve_v7.png"
        fig.savefig(out, dpi=120)
        print("saved", out)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
