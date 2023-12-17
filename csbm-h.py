import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from chi2comb import chi2comb_cdf, ChiSquared


def chi2comb_error(n0, n1, mu0, mu1, sigma0, sigma1):
    df = mu0.shape[0]
    a = 1 / (2 * sigma1) - 1 / (2 * sigma0)
    b = mu0 / sigma0 - mu1 / sigma1
    c = mu1 @ np.transpose(mu1) / (2 * sigma1) - mu0 @ np.transpose(mu0) / (2 * sigma0) + np.log(
        n0 / n1 * (sigma1 / sigma0))

    # generalized chi-square version 2 (put weight to noncentral chi-square)
    gcoef0 = 0  # Coefficient of the standard Normal distribution.
    ncents0 = [(mu0 / np.sqrt(sigma0) + (b) / (2 * a * np.sqrt(sigma0))) @ np.transpose(mu0 / np.sqrt(sigma0) + (b) / (
            2 * a * np.sqrt(sigma0)))]  # noncentrality parameters lambda of the non-centric chi-square variables
    q0 = -c + (b @ np.transpose(b)) / (4 * a)  # quantity for cdf
    dofs0 = [df]  # degree of freedom of the non-centric chi-square variables
    coefs0 = [a * sigma0]  # coefficients of the non-centric chi-square variables
    chi2s0 = [ChiSquared(coefs0[i], ncents0[i], dofs0[i]) for i in
              range(1)]  # List of ChiSquared objects defining noncentral χ² distributions.
    error_x0, errno, info = chi2comb_cdf(q0, chi2s0, gcoef0)

    gcoef1 = 0
    ncents1 = [(mu1 / np.sqrt(sigma1) + (b) / (2 * a * np.sqrt(sigma1))) @ np.transpose(
        mu1 / np.sqrt(sigma1) + (b) / (2 * a * np.sqrt(sigma1)))]
    q1 = -c + (b @ np.transpose(b)) / (4 * a)
    dofs1 = [df]
    coefs1 = [a * sigma1]
    chi2s1 = [ChiSquared(coefs1[i], ncents1[i], dofs1[i]) for i in range(1)]
    error_x1, errno, info = chi2comb_cdf(q1, chi2s1, gcoef1)
    error_x1 = 1 - error_x1

    return error_x0, error_x1


def PBE(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, h):
    error_x0, error_x1 = chi2comb_error(n0, n1, mu0, mu1, sigma0, sigma1)

    error_x = n0 / (n0 + n1) * error_x0 + n1 / (n0 + n1) * (error_x1)

    # h information with node homophily information
    muh0 = h * (mu0 - mu1) + mu1
    sigmah0 = (h * (sigma0 - sigma1) + sigma1) / d0
    muh1 = h * (mu1 - mu0) + mu0
    sigmah1 = (h * (sigma1 - sigma0) + sigma0) / d1
    error_h0, error_h1 = chi2comb_error(n0, n1, muh0, muh1, sigmah0, sigmah1)
    error_h = n0 / (n0 + n1) * error_h0 + n1 / (n0 + n1) * (error_h1)

    muh0_high = (1 - h) * (mu0 - mu1)
    sigmah0_high = (1 + h / d0) * sigma0 + (1 - h) / d0 * sigma1
    muh1_high = (1 - h) * (mu1 - mu0)
    sigmah1_high = (1 + h / d1) * sigma1 + (1 - h) / d1 * sigma0
    error_h0_high, error_h1_high = chi2comb_error(n0, n1, muh0_high, muh1_high, sigmah0_high, sigmah1_high)
    error_h_high = n0 / (n0 + n1) * error_h0_high + n1 / (n0 + n1) * (error_h1_high)

    return error_x, error_h, error_h_high


def Negative_Wasserstein(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, h):
    dim = mu0.shape[0]

    x_nswd = -np.linalg.norm(mu0 - mu1, 2) ** 2 - dim * (np.sqrt(sigma0) - np.sqrt(sigma1)) ** 2

    muh0 = h * (mu0 - mu1) + mu1
    sigmah0 = (h * (sigma0 - sigma1) + sigma1) / d0
    muh1 = h * (mu1 - mu0) + mu0
    sigmah1 = (h * (sigma1 - sigma0) + sigma0) / d1
    h_nswd = -np.linalg.norm(muh0 - muh1, 2) ** 2 - dim * (np.sqrt(sigmah0) - np.sqrt(sigmah1)) ** 2

    muh0_high = (1 - h) * (mu0 - mu1)
    sigmah0_high = (1 + h / d0) * sigma0 + (1 - h) / d0 * sigma1
    muh1_high = (1 - h) * (mu1 - mu0)
    sigmah1_high = (1 + h / d1) * sigma1 + (1 - h) / d1 * sigma0
    h_high_nswd = -np.linalg.norm(muh0_high - muh1_high, 2) ** 2 - dim * (
            np.sqrt(sigmah0_high) - np.sqrt(sigmah1_high)) ** 2

    return x_nswd, h_nswd, h_high_nswd


def Negative_Hellinger(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, h):
    dim = mu0.shape[0]

    x_nshd = -1 + ((np.sqrt(sigma0) * np.sqrt(sigma1)) / ((sigma0 + sigma1) / 2)) ** (dim / 2) * np.exp(
        -1 / 8 * np.linalg.norm(mu0 - mu1, 2) ** 2 / ((sigma0 + sigma1) / 2))

    muh0 = h * (mu0 - mu1) + mu1
    sigmah0 = (h * (sigma0 - sigma1) + sigma1) / d0
    muh1 = h * (mu1 - mu0) + mu0
    sigmah1 = (h * (sigma1 - sigma0) + sigma0) / d1
    h_nshd = -1 + ((np.sqrt(sigmah0) * np.sqrt(sigmah1)) / ((sigmah0 + sigmah1) / 2)) ** (dim / 2) * np.exp(
        -1 / 8 * np.linalg.norm(muh0 - muh1, 2) ** 2 / ((sigmah0 + sigmah1) / 2))

    muh0_high = (1 - h) * (mu0 - mu1)
    sigmah0_high = (1 + h / d0) * sigma0 + (1 - h) / d0 * sigma1
    muh1_high = (1 - h) * (mu1 - mu0)
    sigmah1_high = (1 + h / d1) * sigma1 + (1 - h) / d1 * sigma0
    h_high_nshd = -1 + ((np.sqrt(sigmah0_high) * np.sqrt(sigmah1_high)) / ((sigmah0_high + sigmah1_high) / 2)) ** (
            dim / 2) * np.exp(
        -1 / 8 * np.linalg.norm(muh0_high - muh1_high, 2) ** 2 / ((sigmah0_high + sigmah1_high) / 2))

    return x_nshd, h_nshd, h_high_nshd


def NGJ_div(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, h):
    dim = mu0.shape[0]
    p0, p1 = n0 / (n0 + n1), n1 / (n0 + n1)
    rho = np.sqrt(sigma0 / sigma1)
    dE2 = (mu0 - mu1).T @ (mu0 - mu1)
    NGJD_x = dE2 / 2 * (p0 / sigma1 + p1 / sigma0) + dim / 2 * (p0 * rho ** 2 + p1 / (rho ** 2) - 1)
    dist_x = dE2 / 2 * (p0 / sigma1 + p1 / sigma0)

    muh0 = h * (mu0 - mu1) + mu1
    sigmah0 = (h * (sigma0 - sigma1) + sigma1) / d0
    muh1 = h * (mu1 - mu0) + mu0
    sigmah1 = (h * (sigma1 - sigma0) + sigma0) / d1
    rhoh = np.sqrt(sigmah0 / sigmah1)
    dE2h = (muh0 - muh1).T @ (muh0 - muh1)
    NGJD_h = dE2h / 2 * (p0 / sigmah1 + p1 / sigmah0) + dim / 2 * (p0 * rhoh ** 2 + p1 / (rhoh ** 2) - 1)
    dist_h = dE2h / 2 * (p0 / sigmah1 + p1 / sigmah0)

    muh0_high = (1 - h) * (mu0 - mu1)
    sigmah0_high = (1 + h / d0) * sigma0 + (1 - h) / d0 * sigma1
    muh1_high = (1 - h) * (mu1 - mu0)
    sigmah1_high = (1 + h / d1) * sigma1 + (1 - h) / d1 * sigma0
    rho_high = np.sqrt(sigmah0_high / sigmah1_high)
    dE2_high = (muh0_high - muh1_high).T @ (muh0_high - muh1_high)
    NGJD_high = dE2_high / 2 * (p0 / sigmah1_high + p1 / sigmah0_high) + dim / 2 * (
            p0 * rho_high ** 2 + p1 / (rho_high ** 2) - 1)
    dist_high = dE2_high / 2 * (p0 / sigmah1_high + p1 / sigmah0_high)

    return -NGJD_x, -NGJD_h, -NGJD_high, - dist_x, -dist_h, -dist_high


def visualize_csbmh(cluster0, cluster1, title, file, filename):
    plt.plot(cluster0[0], cluster0[1], 'x')
    plt.plot(cluster1[0], cluster1[1], 'x')

    plt.axis('equal')
    plt.title(title)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    Path(f"./csbmh_plots/visualization/{file}").mkdir(parents=True, exist_ok=True)
    fig_name = f"./csbmh_plots/visualization/{file}/{filename}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()


def csbm_h_2(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, return_results=False):
    n_batch = 1

    homo_range = np.linspace(0, 1, 101)
    pbe_x_results, pbe_h_results, pbe_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))
    Dist_x_results, Dist_h_results, Dist_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))
    NGJD_x_results, NGJD_h_results, NGJD_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))
    ratio_x_results, ratio_h_results, ratio_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))
    nswd_x_results, nswd_h_results, nswd_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))
    nshd_x_results, nshd_h_results, nshd_h_high_results = np.zeros((homo_range.shape[0], n_batch)), np.zeros(
        (homo_range.shape[0], n_batch)), np.zeros((homo_range.shape[0], n_batch))

    visualize = 0
    for i, h in zip(range(homo_range.shape[0]), homo_range):
        for j in range(n_batch):
            x0 = np.random.multivariate_normal(mu0, sigma0 * np.eye(2), n0).T
            x1 = np.random.multivariate_normal(mu1, sigma1 * np.eye(2), n1).T

            muh0 = h * (mu0 - mu1) + mu1
            sigmah0 = (h * (sigma0 - sigma1) + sigma1) / d0
            muh1 = h * (mu1 - mu0) + mu0
            sigmah1 = (h * (sigma1 - sigma0) + sigma0) / d1

            muh0_high = (1 - h) * (mu0 - mu1)
            sigmah0_high = (1 + h / d0) * sigma0 + (1 - h) / d0 * sigma1
            muh1_high = (1 - h) * (mu1 - mu0)
            sigmah1_high = (1 + h / d1) * sigma1 + (1 - h) / d1 * sigma0

            h0 = np.random.multivariate_normal(muh0, sigmah0 * np.eye(2), n0).T
            h1 = np.random.multivariate_normal(muh1, sigmah1 * np.eye(2), n1).T

            h0_high = np.random.multivariate_normal(muh0_high, sigmah0_high * np.eye(2), n0).T
            h1_high = np.random.multivariate_normal(muh1_high, sigmah1_high * np.eye(2), n1).T

            pbe_x_results[i, j], pbe_h_results[i, j], pbe_h_high_results[i, j] = PBE(n0, n1, mu0, mu1, sigma0, sigma1,
                                                                                     d0, d1, h)
            NGJD_x_results[i, j], NGJD_h_results[i, j], NGJD_h_high_results[i, j], Dist_x_results[i, j], Dist_h_results[
                i, j], Dist_h_high_results[i, j] = NGJ_div(n0, n1, mu0, mu1, sigma0, sigma1, d0, d1, h)
            ratio_x_results[i, j], ratio_h_results[i, j], ratio_h_high_results[i, j] = NGJD_x_results[i, j] - \
                                                                                       Dist_x_results[i, j], \
                                                                                       NGJD_h_results[i, j] - \
                                                                                       Dist_h_results[i, j], \
                                                                                       NGJD_h_high_results[i, j] - \
                                                                                       Dist_h_high_results[i, j]

            nswd_x_results[i, j], nswd_h_results[i, j], nswd_h_high_results[i, j] = Negative_Wasserstein(n0, n1, mu0,
                                                                                                         mu1, sigma0,
                                                                                                         sigma1, d0, d1,
                                                                                                         h)
            nshd_x_results[i, j], nshd_h_results[i, j], nshd_h_high_results[i, j] = Negative_Hellinger(n0, n1, mu0, mu1,
                                                                                                       sigma0, sigma1,
                                                                                                       d0, d1, h)

            if visualize == 1:
                if j == 0:
                    visualize_csbmh(h0, h1, 'homophily %.2f' % h,
                                    f'n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}',
                                    'homophily=%.2f' % h)
                    visualize_csbmh(h0_high, h1_high, 'homophily %.2f' % h,
                                    f'n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}',
                                    'homophily=%.2f' % h)

                if j == 0 and i == 100:
                    visualize_csbmh(x0, x1, 'original features',
                                    f'n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}',
                                    'original feature')

    if return_results:
        return np.mean(pbe_x_results, 1), np.mean(pbe_h_results, 1), np.mean(pbe_h_high_results, 1), np.mean(
            NGJD_x_results, 1), np.mean(NGJD_h_results, 1), np.mean(NGJD_h_high_results, 1)

    # Plot Probabilistic Bayes Error Rate
    plt.plot(homo_range, np.mean(pbe_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(pbe_x_results, 1) + np.std(pbe_x_results, 1),
                     np.mean(pbe_x_results, 1) - np.std(pbe_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(pbe_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(pbe_h_results, 1) + np.std(pbe_h_results, 1),
                     np.mean(pbe_h_results, 1) - np.std(pbe_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(pbe_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(pbe_h_high_results, 1) + np.std(pbe_h_high_results, 1),
                     np.mean(pbe_h_high_results, 1) - np.std(pbe_h_high_results, 1), color='red', alpha=0.35)

    plt.legend(["x error", 'h error', 'h_high error'], bbox_to_anchor=(1, 1))
    plt.title('Probabilistic Bayes Error')
    plt.xlim(0, 1)
    plt.ylim(0, 0.55)
    plt.xlabel('Homophily h')

    sub_dir_names = ("x_data", "h_data", "h_high_data")
    for sub_dir_name in sub_dir_names:
        Path(f"./csbmh_plots/data/{sub_dir_name}").mkdir(parents=True, exist_ok=True)

    np.save(
        f"./csbmh_plots/data/x_data/Probabilistic_Error,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        pbe_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/Probabilistic_Error,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        pbe_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/Probabilistic_Error,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        pbe_h_high_results)

    fig_name = f"./csbmh_plots/Probabilistic_Error,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot weighted Euclidean distance
    plt.plot(homo_range, np.mean(Dist_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(Dist_x_results, 1) + np.std(Dist_x_results, 1),
                     np.mean(Dist_x_results, 1) - np.std(Dist_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(Dist_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(Dist_h_results, 1) + np.std(Dist_h_results, 1),
                     np.mean(Dist_h_results, 1) - np.std(Dist_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(Dist_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(Dist_h_high_results, 1) + np.std(Dist_h_high_results, 1),
                     np.mean(Dist_h_high_results, 1) - np.std(Dist_h_high_results, 1), color='red', alpha=0.35)

    plt.legend(["X", 'H', 'H_high'], bbox_to_anchor=(1, 1))
    plt.title('Expected Negative Normalized Euclidean Distance')
    plt.xlim(0, 1)
    plt.xlabel('Homophily h')
    Path(f"./csbmh_plots/data").mkdir(parents=True, exist_ok=True)
    np.save(
        f"./csbmh_plots/data/x_data/WE_dist,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        Dist_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/WE_dist,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        Dist_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/WE_dist,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        Dist_h_high_results)

    fig_name = f"./csbmh_plots/WE_dist,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Negative Generalized Jeffreys divergence
    plt.plot(homo_range, np.mean(NGJD_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(NGJD_x_results, 1) + np.std(NGJD_x_results, 1),
                     np.mean(NGJD_x_results, 1) - np.std(NGJD_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(NGJD_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(NGJD_h_results, 1) + np.std(NGJD_h_results, 1),
                     np.mean(NGJD_h_results, 1) - np.std(NGJD_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(NGJD_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(NGJD_h_high_results, 1) + np.std(NGJD_h_high_results, 1),
                     np.mean(NGJD_h_high_results, 1) - np.std(NGJD_h_high_results, 1), color='red', alpha=0.35)

    plt.legend(["X", 'H', 'H_high'], bbox_to_anchor=(1, 1))
    plt.title('Negative Generalized Jeffreys Divergence')
    plt.xlim(0, 1)
    # plt.ylim(0,0.5)
    plt.xlabel('Homophily h')
    Path(f"./csbmh_plots/data").mkdir(parents=True, exist_ok=True)
    np.save(
        f"./csbmh_plots/data/x_data/NGJD,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        NGJD_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/NGJD,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        NGJD_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/NGJD,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        NGJD_h_high_results)

    fig_name = f"./csbmh_plots/NGJD,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot the  ratio(rho) term in KL-divergence
    plt.plot(homo_range, np.mean(ratio_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(ratio_x_results, 1) + np.std(ratio_x_results, 1),
                     np.mean(ratio_x_results, 1) - np.std(ratio_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(ratio_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(ratio_h_results, 1) + np.std(ratio_h_results, 1),
                     np.mean(ratio_h_results, 1) - np.std(ratio_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(ratio_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(ratio_h_high_results, 1) + np.std(ratio_h_high_results, 1),
                     np.mean(ratio_h_high_results, 1) - np.std(ratio_h_high_results, 1), color='red', alpha=0.35)
    plt.legend(["X", 'H', 'H_high'], bbox_to_anchor=(1, 1))
    plt.title('Negative Variance Ratio')
    plt.xlim(0, 1)
    plt.xlabel('Homophily h')
    Path(f"./csbmh_plots/data").mkdir(parents=True, exist_ok=True)
    np.save(
        f"./csbmh_plots/data/x_data/ratio,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/ratio,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/ratio,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_high_results)

    fig_name = f"./csbmh_plots/ratio,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot the  Negative Squared Wasserstein Distance
    plt.plot(homo_range, np.mean(nswd_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(nswd_x_results, 1) + np.std(nswd_x_results, 1),
                     np.mean(nswd_x_results, 1) - np.std(nswd_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(nswd_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(nswd_h_results, 1) + np.std(nswd_h_results, 1),
                     np.mean(nswd_h_results, 1) - np.std(nswd_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(nswd_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(nswd_h_high_results, 1) + np.std(nswd_h_high_results, 1),
                     np.mean(nswd_h_high_results, 1) - np.std(nswd_h_high_results, 1), color='red', alpha=0.35)
    plt.legend(["X", 'H', 'H_high'], bbox_to_anchor=(1, 1))
    plt.title('Negative Squared Wasserstein Distance')
    plt.xlim(0, 1)
    plt.xlabel('Homophily h')
    Path(f"./csbmh_plots/data").mkdir(parents=True, exist_ok=True)
    np.save(
        f"./csbmh_plots/data/x_data/nswd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/nswd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/nswd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_high_results)

    fig_name = f"./csbmh_plots/nswd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot the  Negative Squared Hellinger Distance
    plt.plot(homo_range, np.mean(nshd_x_results, 1), color='black')
    plt.fill_between(homo_range, np.mean(nshd_x_results, 1) + np.std(nshd_x_results, 1),
                     np.mean(nshd_x_results, 1) - np.std(nshd_x_results, 1), color='black', alpha=0.35)
    plt.plot(homo_range, np.mean(nshd_h_results, 1), color='green')
    plt.fill_between(homo_range, np.mean(nshd_h_results, 1) + np.std(nshd_h_results, 1),
                     np.mean(nshd_h_results, 1) - np.std(nshd_h_results, 1), color='green', alpha=0.35)
    plt.plot(homo_range, np.mean(nshd_h_high_results, 1), color='red')
    plt.fill_between(homo_range, np.mean(nshd_h_high_results, 1) + np.std(nshd_h_high_results, 1),
                     np.mean(nshd_h_high_results, 1) - np.std(nshd_h_high_results, 1), color='red', alpha=0.35)
    plt.legend(["X", 'H', 'H_high'], bbox_to_anchor=(1, 1))
    plt.title('Negative Squared Hellinger  Distance')
    plt.xlim(0, 1)
    plt.xlabel('Homophily h')
    Path(f"./csbmh_plots/data").mkdir(parents=True, exist_ok=True)
    np.save(
        f"./csbmh_plots/data/x_data/nshd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_x_results)
    np.save(
        f"./csbmh_plots/data/h_data/nshd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_results)
    np.save(
        f"./csbmh_plots/data/h_high_data/nshd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}",
        ratio_h_high_results)

    fig_name = f"./csbmh_plots/nshd,n0={n0},n1={n1},mu0x={mu0[0]},mu1x={mu1[0]},sigma0={sigma0},sigma1={sigma1},d0={d0},d1={d1}.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()


def len_eq_2(input):
    if len(input) == 2 and all([type(el) is int for el in input]):
        return input
    raise argparse.ArgumentTypeError("Parameter must be 2 integers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prior_distribution_params",
                        default=[100, 100], nargs='+', type=len_eq_2,
                        help="Parameters (n0, n1) to set prior distributions")
    parser.add_argument("--node_center_0",
                        default=[-1, 0], nargs="+", type=len_eq_2,
                        help="Node centers mu0 for ablation on inter-class node distinguishability")
    parser.add_argument("--node_center_1",
                        default=[1, 0], nargs="+", type=len_eq_2,
                        help="Node centers mu1 for ablation on inter-class node distinguishability")
    parser.add_argument("--sigmas",
                        default=[1, 5], nargs="+", type=len_eq_2,
                        help="Sigma0 and sigma1 for ablation on inter-class node distinguishability")
    parser.add_argument("--node_degrees",
                        default=[5, 5], nargs="+", type=len_eq_2,
                        help="Node degree d0 and d1 for ablation on inter-class node distinguishability")
    args = parser.parse_args()

    # 2D plots of CSBM-H
    n0, n1 = args.prior_distribution_params
    mu0 = np.array(args.node_center_0)
    mu1 = np.array(args.node_center_1)
    sigma0, sigma1 = args.sigmas
    d0, d1 = args.node_degrees

    n0_range = np.linspace(100, 500, 1)
    n1_range = np.linspace(100, 1000, 10)
    mu0x_range = np.linspace(1, 5, 5)
    mu1x_range = np.linspace(1, 5, 5)
    sigma0_range = np.linspace(1, 21, 5)
    sigma1_range = np.linspace(1, 5, 1)
    d0_range = np.linspace(5, 55, 6)
    d1_range = np.linspace(5, 15, 1)

    csbm_h_2(int(n0), int(n1), mu0, mu1, sigma0, sigma1, d0, d1)
