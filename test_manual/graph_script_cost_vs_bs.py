import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, os.path

def prettify(optim_type):
    if optim_type == 'profile_both':
        return 'Profile Compute and Memory'
    elif optim_type == 'profile_comp':
        return 'Profile Compute Only'
    elif optim_type == 'profile_mem':
        return 'Profile Memory Only'
    elif optim_type == 'uniform_both':
        return 'Uniform Compute and Memory'
    else:
        print('Called prettify(optim_type=%s) with unknown optim_type.' % optim_type)
        return optim_type


optim_types = ['profile_both', 'uniform_both']

results = {'profile_both': {'bs': [8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
                  'peak': [436.0,
                           644.0,
                           1180.0,
                           1740.0,
                           2280.0,
                           2828.0,
                           3368.0,
                           3180.0,
                           3632.0,
                           3608.0,
                           4004.0],
                  'time': [116.1702852702389,
                           172.74786718562245,
                           316.01457092848915,
                           437.299822592487,
                           554.3008846174926,
                           679.2142285897087,
                           801.7515305901567,
                           1086.0161230048784,
                           1276.1014533235382,
                           1549.6443236699947,
                           2576.35956104286]},
 'uniform_both': {'bs': [8,
                         16,
                         32,
                         48,
                         64,
                         80,
                         96,
                         112,
                         128,
                         144,
                         160,
                         176,
                         192,
                         208],
                  'peak': [436.0,
                           644.0,
                           1180.0,
                           1740.0,
                           2280.0,
                           2828.0,
                           3368.0,
                           3924.0,
                           4468.0,
                           5028.0,
                           5400.0,
                           5660.0,
                           6172.0,
                           6252.0],
                  'time': [116.27488707626858,
                           172.77136276351905,
                           318.83357078147424,
                           444.4604597706348,
                           556.3128771117576,
                           689.3480091088763,
                           805.6905463555208,
                           1035.6567938886583,
                           1177.262846049542,
                           1322.5290366752695,
                           1493.4633639082308,
                           1670.8913431155186,
                           1828.4614942359426,
                           2401.819091714919]}
}

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
ax1 = axes[0]
ax2 = axes[1]
fmts = {
    'profile_both': 'g-',
    'uniform_both': 'm--'
}

ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Simulated Time (ms)')
for optim_type in optim_types:
    batch_sizes = results[optim_type]['bs']#[:10]
    times = results[optim_type]['time']#[:10]
    print(times)
    fmt = fmts[optim_type]
    ax1.plot(batch_sizes, times, fmt, label=prettify(optim_type))
ax1.legend()

# ax1.set_xlabel('Batch Size')
# ax1.set_ylabel('Simulated Time (ms)')
# for optim_type in optim_types:
#     fmt = fmts[optim_type]
#     label = prettify(optim_type)
#     if optim_type == 'profile_both': optim_type = 'uniform_both'
#     elif optim_type == 'uniform_both': optim_type = 'profile_both'
#     batch_sizes = results[optim_type]['bs']#[:10]
#     times = results[optim_type]['time']#[:10]
#     print(times)
#     ax1.plot(batch_sizes, times, fmt, label=label)
# ax1.legend()

ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Simulated Peak Memory (MB)')
for optim_type in optim_types:
    batch_sizes = results[optim_type]['bs']#[:10]
    peaks = results[optim_type]['peak']#[:10]
    print(peaks)
    fmt = fmts[optim_type]
    ax2.plot(batch_sizes, peaks, fmt, label=prettify(optim_type))
ax2.legend()

outfile_path = os.path.join("results", "cost_vs_batch_size_SOMETHING.eps")
plt.savefig(outfile_path)
