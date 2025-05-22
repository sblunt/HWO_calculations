import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import emcee
import os
import sys

random_seed = 1

np.random.seed(random_seed)

## variables: sigma_true, n_stars, age_unc
## make one plot for sigma_true = 1, one plot for sigma_true = 0.1

"""
assume a true distribution of oxygen-onset times
"""
mu_true = float(sys.argv[1])  # [Gyr]

sigma_true = float(sys.argv[2])  # [Gyr]

"""
draw a dataset
"""

n_stars = int(sys.argv[3])
age_unc_percent = float(sys.argv[4])  # percentage


working_dir = "results/mu{}_sigma{}_ageunc{}%_Nstars{}".format(
    mu_true, sigma_true, int(100 * age_unc_percent), n_stars
)
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

# draw a sample of true ages
true_ages = np.random.uniform(0, 13, size=n_stars)

# add Gaussian noise
sampled_ages = true_ages + np.random.normal(
    scale=age_unc_percent * true_ages, size=n_stars
)
age_uncertainties = np.ones_like(sampled_ages) * age_unc_percent * true_ages

# assign each star a "detection" or "nondetection" of O3 based on the assumed
# true distribution
random_comparions = np.random.normal(mu_true, sigma_true, size=n_stars)
o3_detections = true_ages > random_comparions

# plot the true distribution & the dataset
ages2plot = np.linspace(0, 13, int(1e4))

fig, ax = plt.subplots(2, 1, sharex=True)
# plt.subplots_adjust(hspace=0)
for i, a in enumerate(ax):
    a.plot(
        ages2plot,
        norm.pdf(ages2plot, mu_true, sigma_true),
        color="k",
        lw=4,
        zorder=20,
        alpha=0.5,
    )
    a.set_ylabel("prob.")
ax[0].set_title("detections")
ax[1].set_title("nondetections")

for i, measured_age in enumerate(sampled_ages):
    if o3_detections[i]:
        ls = "-"
        a = ax[0]
    else:
        ls = "-"
        a = ax[1]
    a.scatter(
        true_ages[i],
        norm.pdf(true_ages[i], measured_age, age_unc_percent * true_ages[i])
        / np.max(norm.pdf(ages2plot, measured_age, age_unc_percent * true_ages[i]))
        * np.max(norm.pdf(ages2plot, mu_true, sigma_true)),
        marker="*",
        color="rebeccapurple",  # plt.get_cmap("prism")(i / n_stars),
        # alpha=0.2,
    )
    a.plot(
        ages2plot,
        norm.pdf(ages2plot, measured_age, age_unc_percent * true_ages[i])
        / np.max(norm.pdf(ages2plot, measured_age, age_unc_percent * true_ages[i]))
        * np.max(norm.pdf(ages2plot, mu_true, sigma_true)),
        color="rebeccapurple",  # plt.get_cmap("prism")(i / n_stars),
        ls=ls,
        alpha=0.2,
    )
plt.xlabel("age [Gyr]")
plt.savefig("{}/seed{}_sample.png".format(working_dir, random_seed), dpi=250)

"""
define the likelihood
"""


def single_star_loglike(params_arr, star_age, star_age_unc, star_meas_o3):
    mu_pop = params_arr[0]
    sigma_pop = params_arr[1]

    # O3 measured in star: we want the prob. that we would measure O3
    # in this star given the proposal distribution and the age unc.
    if star_meas_o3:
        loglike = norm.logcdf(
            0, loc=mu_pop - star_age, scale=np.sqrt(sigma_pop**2 + star_age_unc**2)
        )

    # no O3 measured (opposite of above)
    else:
        loglike = norm.logcdf(
            0,
            loc=star_age - mu_pop,
            scale=np.sqrt(sigma_pop**2 + star_age_unc**2),
        )

    return loglike


def loglike(params_arr, star_ages_arr, star_age_unc_arr, star_meas_o3_arr):
    n_stars = len(star_ages_arr)
    loglike = 0
    for i in range(n_stars):
        loglike += single_star_loglike(
            params_arr, star_ages_arr[i], star_age_unc_arr[i], star_meas_o3_arr[i]
        )

    return loglike


def logposterior(params_arr, star_ages_arr, star_age_unc_arr, star_meas_o3_arr):
    mu_pop = params_arr[0]
    sigma_pop = params_arr[1]

    # set priors on pop-level parameters
    if mu_pop < 0 or mu_pop > 13:
        return -np.inf
    if sigma_pop < 0 or sigma_pop > 13:
        return -np.inf

    total_loglike = loglike(
        params_arr, star_ages_arr, star_age_unc_arr, star_meas_o3_arr
    )
    if np.isnan(total_loglike):
        return -np.inf

    return total_loglike


"""
run mcmc!
"""

ndim, nwalkers = 2, 100
n_burn = 200
n_prod = 500

p0 = np.random.uniform(0, 13, size=(nwalkers, ndim))
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, logposterior, args=(sampled_ages, age_uncertainties, o3_detections)
)

# print("running burn in!")
# state = sampler.run_mcmc(p0, n_burn)
# sampler.reset()
# print("running production chain!")
# sampler.run_mcmc(p0, n_prod)
# print("done")

# samples = sampler.flatchain
# np.save(f"{working_dir}/seed{random_seed}_chains.npy", samples)

samples = np.load(f"{working_dir}/seed{random_seed}_chains.npy")
mu_samples = samples[:, 0]
sigma_samples = samples[:, 1]

"""
plot results
"""

fig, ax = plt.subplots(2, 1)
ax[0].hist(
    mu_samples,
    bins=50,
    color="rebeccapurple",
    alpha=0.5,
    label="recovered posterior",
    density=True,
)
ax[0].axvline(mu_true, color="k", label="true value")
ax[0].set_xlabel("$\\mu_{\\mathrm{{pop}}}$ [Gyr]")
ax[1].set_xlabel(("$\\sigma_{\\mathrm{{pop}}}$ [Gyr]"))
ax[1].hist(sigma_samples, bins=50, color="rebeccapurple", alpha=0.5, density=True)
ax[1].axvline(sigma_true, color="k")
ax[0].set_ylabel("prob.")
ax[1].set_ylabel("prob.")


mu_cis = np.quantile(mu_samples, [0.16, 0.84])
sigma_cis = np.quantile(sigma_samples, [0.16, 0.84])
# ax[0].axvline(mu_cis[0], alpha=0.5, ls="--", color="k", label="$\mu$ precision")
# ax[0].axvline(mu_cis[1], alpha=0.5, ls="--", color="k")

plt.sca(ax[0])
plt.annotate(
    "",
    xy=(mu_cis[0], 0.5),
    xytext=(mu_cis[1], 0.5),
    arrowprops=dict(arrowstyle="<->"),
)
plt.sca(ax[1])
plt.annotate(
    "",
    xy=(sigma_cis[0], 0.5),
    xytext=(sigma_cis[1], 0.5),
    arrowprops=dict(arrowstyle="<->"),
)
ax[0].legend()

# ax[1].axvspan(sigma_cis[0], sigma_cis[1], alpha=0.5, color="purple")
plt.tight_layout()
plt.savefig("{}/seed{}_recovery.png".format(working_dir, random_seed), dpi=250)

"""
plot chains to assess convergence
"""

fig, ax = plt.subplots(2, 1)

mu_reshaped = mu_samples.reshape((nwalkers, n_prod))
sigma_reshaped = sigma_samples.reshape((nwalkers, n_prod))

for i in range(nwalkers):
    ax[0].plot(mu_reshaped[i, :], color="k", alpha=0.01)
    ax[1].plot(sigma_reshaped[i, :], color="k", alpha=0.01)
plt.savefig("{}/seed{}_chains.png".format(working_dir, random_seed), dpi=250)
