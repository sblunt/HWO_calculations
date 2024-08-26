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
mu_true = 5  # [Gyr]

sigma_true = float(sys.argv[1])  # [Gyr]

"""
draw a dataset
"""

n_stars = int(sys.argv[2])
age_unc = float(sys.argv[3])  # [Gyr]


working_dir = "results/mu{}_sigma{}_ageunc{}_Nstars{}".format(
    mu_true, sigma_true, age_unc, n_stars
)
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

# draw a sample of true ages
true_ages = np.random.uniform(0, 13, size=n_stars)

# add Gaussian noise
sampled_ages = true_ages + np.random.normal(scale=age_unc, size=n_stars)
age_uncertainties = np.ones_like(sampled_ages) * age_unc

# assign each star a "detection" or "nondetection" of O3 based on the assumed
# true distribution
random_comparions = np.random.normal(mu_true, sigma_true, size=n_stars)
o3_detections = sampled_ages > random_comparions

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
        norm.pdf(true_ages[i], measured_age, age_unc)
        / np.max(norm.pdf(ages2plot, measured_age, age_unc))
        * np.max(norm.pdf(ages2plot, mu_true, sigma_true)),
        marker="*",
        color=plt.get_cmap("prism")(i / n_stars),
    )
    a.plot(
        ages2plot,
        norm.pdf(ages2plot, measured_age, age_unc)
        / np.max(norm.pdf(ages2plot, measured_age, age_unc))
        * np.max(norm.pdf(ages2plot, mu_true, sigma_true)),
        color=plt.get_cmap("prism")(i / n_stars),
        ls=ls,
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
        # likelihood = prob. that age (characterized by measurement unc. distribution)
        # is greater than population distribution (characterized by proposal distribution).
        # This is the same as the prob. that the difference of the two Gaussians (which is
        # also Gaussian) is greater than 0
        loglike = norm.logcdf(
            0, loc=mu_pop - star_age, scale=np.sqrt(sigma_pop**2 + star_age_unc**2)
        )

    # no O3 measured (opposite of above)
    else:
        loglike = norm.logcdf(
            0, loc=star_age - mu_pop, scale=np.sqrt(sigma_pop**2 + star_age_unc**2)
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

    return loglike(params_arr, star_ages_arr, star_age_unc_arr, star_meas_o3_arr)


"""
run mcmc!
"""

ndim, nwalkers = 2, 100

p0 = np.random.uniform(0, 13, size=(nwalkers, ndim))
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, logposterior, args=(sampled_ages, age_uncertainties, o3_detections)
)

print("running burn in!")
state = sampler.run_mcmc(p0, 200)
sampler.reset()
print("running production chain!")
sampler.run_mcmc(p0, 500)
print("done")

samples = sampler.flatchain
mu_samples = samples[:, 0]
sigma_samples = samples[:, 1]
np.save(f"{working_dir}/seed{random_seed}_chains.npy", samples)

"""
plot results
"""

fig, ax = plt.subplots(2, 1)
ax[0].hist(
    mu_samples, bins=50, color="rebeccapurple", alpha=0.5, label="recovered posterior"
)
ax[0].axvline(mu_true, color="k", label="true value")
ax[0].legend()
ax[0].set_xlabel("$\\mu_{\\mathrm{{population}}}$ [Gyr]")
ax[1].set_xlabel(("$\\sigma_{\\mathrm{{population}}}$ [Gyr]"))
ax[1].hist(sigma_samples, bins=50, color="rebeccapurple", alpha=0.5)
ax[1].axvline(sigma_true, color="k")
plt.tight_layout()
plt.savefig("{}/seed{}_recovery.png".format(working_dir, random_seed), dpi=250)
