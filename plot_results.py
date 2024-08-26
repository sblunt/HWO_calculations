# read in all numpy chain files
# for each value of age uncertainty:
# 1) plot num stars vs uncertainty on onset time
# 2) plot num stars vs uncertainy on onset spread

import numpy as np
import matplotlib.pyplot as plt

seed = 1

mu_true = 2
sigma_true = 0.1  # 2, 0.5, 0.1

n_stars = np.array([15, 25, 50, 60])
age_unc = np.array([0.2, 1, 5])

fig, ax = plt.subplots(2, 1, sharex=True)
plt.subplots_adjust(hspace=0)
ax[0].set_title(
    "Simulation for true oxygen onset dist ${:.1f}\\pm{:.1f}$ Gyr".format(
        mu_true, sigma_true
    )
)
alphas = age_unc / age_unc[-1]
for a, age in enumerate(age_unc):
    mu_unc_array = np.zeros(len(n_stars))
    sigma_unc_array = np.zeros(len(n_stars))
    for i, stars in enumerate(n_stars):
        chains = np.load(
            "results/mu{}_sigma{:.1f}_ageunc{:.1f}_Nstars{}/seed{}_chains.npy".format(
                mu_true, sigma_true, age, stars, seed
            )
        )
        mu_samples = chains[:, 0]
        sigma_samples = chains[:, 1]

        mu_unc_array[i] = np.std(mu_samples)
        sigma_unc_array[i] = np.std(sigma_samples)

    ax[0].plot(
        n_stars,
        mu_unc_array,
        label=f"{age} Gyr age unc.",
        alpha=alphas[a],
        color="rebeccapurple",
    )
    ax[1].plot(n_stars, sigma_unc_array, alpha=alphas[a], color="rebeccapurple")

# add lines of const % precision
ax[0].axhline([mu_true], label="100% precision", color="k", ls="--")
ax[0].axhline([0.5 * mu_true], label="50% precision", color="k", ls="--", alpha=0.1)
ax[0].axhline([0.1 * mu_true], label="10% precision", color="k", ls="--", alpha=0.01)
ax[1].axhline([sigma_true], color="k", ls="--")
ax[1].axhline([0.5 * sigma_true], color="k", ls="--", alpha=0.1)
ax[0].axhline([0.1 * mu_true], color="k", ls="--", alpha=0.01)

ax[0].set_ylabel("$\\mu$ precision [Gyr]")
ax[0].legend()

ax[1].set_ylabel("$\\sigma$ precision [Gyr]")
ax[1].set_xlabel("number of Earth analogs in sample")
plt.savefig(f"results/summary_mu{mu_true}_sigma{sigma_true}.png", dpi=250)
