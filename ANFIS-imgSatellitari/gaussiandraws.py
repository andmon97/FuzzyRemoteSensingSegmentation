import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
x_rgb = np.arange(0, 255, 1)
x_entropy = np.arange(0, 6, 0.001)

# Generate fuzzy membership functions
#red
rmf0 = fuzz.membership.gaussmf(x_rgb, -90.43, 118.37)
rmf1 = fuzz.membership.gaussmf(x_rgb, 183.36, 118.92)
#green
gmf0 = fuzz.membership.gaussmf(x_rgb, -68.45, 69.85)
gmf1 = fuzz.membership.gaussmf(x_rgb,226.86, 110.04)
#blue
bmf0 = fuzz.membership.gaussmf(x_rgb, -82.20, 113.18)
bmf1 = fuzz.membership.gaussmf(x_rgb,204.32, 124.37)
#entropy
emf0 = fuzz.membership.gaussmf(x_entropy, 1.008, 2.104)
emf1 = fuzz.membership.gaussmf(x_entropy, 5.189, 1.576)

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(4, 6))

ax0.plot(x_rgb, rmf0, 'r', linewidth=1.5, label='Low ')
ax0.plot(x_rgb, rmf1, 'g', linewidth=1.5, label='High ')
ax0.set_title('Red')


ax1.plot(x_rgb, gmf0, 'r', linewidth=1.5 )
ax1.plot(x_rgb, gmf1, 'g', linewidth=1.5)
ax1.set_title('Green')

ax2.plot(x_rgb, bmf0, 'r', linewidth=1.5)
ax2.plot(x_rgb, bmf1, 'g', linewidth=1.5)
ax2.set_title('Blue')

ax3.plot(x_entropy, emf0, 'r', linewidth=1.5)
ax3.plot(x_entropy, emf1, 'g', linewidth=1.5)
ax3.set_title('Entropy')
ax3.legend([emf0, emf1], ['Low', 'High'])

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

#-----------------------------------------------------------#
# Generate fuzzy membership functions
#red
rmf0 = fuzz.membership.gaussmf(x_rgb, 3.179, 34.25)
rmf1 = fuzz.membership.gaussmf(x_rgb, 98.119, 45.162)
rmf2 = fuzz.membership.gaussmf(x_rgb, 176.19, 51.39)
#green
gmf0 = fuzz.membership.gaussmf(x_rgb, -18.452, 58.887)
gmf1 = fuzz.membership.gaussmf(x_rgb,100.268, 50.43)
gmf2 = fuzz.membership.gaussmf(x_rgb,221.711, 71.398)
#blue
bmf0 = fuzz.membership.gaussmf(x_rgb, -40.786, 50.649)
bmf1 = fuzz.membership.gaussmf(x_rgb,107.33, 61.56)
bmf2 = fuzz.membership.gaussmf(x_rgb,194.326, 61.950)
#entropy
emf0 = fuzz.membership.gaussmf(x_entropy, 1.749,0.929)
emf1 = fuzz.membership.gaussmf(x_entropy, 3.415, 0.7542)
emf2 = fuzz.membership.gaussmf(x_entropy, 4.80, 0.591)

# Visualize these universes and membership functions
fig2, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(4, 6))

ax0.plot(x_rgb, rmf0, 'r', linewidth=1.5, label='Low')
ax0.plot(x_rgb, rmf1, 'b', linewidth=1.5, label='Medium')
ax0.plot(x_rgb, rmf2, 'g', linewidth=1.5, label='High')
ax0.set_title('Red')
ax0.legend()

ax1.plot(x_rgb, gmf0, 'r', linewidth=1.5)
ax1.plot(x_rgb, gmf1, 'b', linewidth=1.5)
ax1.plot(x_rgb, gmf2, 'g', linewidth=1.5)
ax1.set_title('Green')
ax1.legend()

ax2.plot(x_rgb, bmf0, 'r', linewidth=1.5)
ax2.plot(x_rgb, bmf1, 'b', linewidth=1.5)
ax2.plot(x_rgb, bmf2, 'g', linewidth=1.5)
ax2.set_title('Blue')
ax2.legend()

ax3.plot(x_entropy, emf0, 'r', linewidth=1.5)
ax3.plot(x_entropy, emf1, 'b', linewidth=1.5)
ax3.plot(x_entropy, emf2, 'g', linewidth=1.5)
ax3.set_title('Entropy')
ax3.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

#-----------------------------------------------------------#
# Generate fuzzy membership functions
#red
rmf0 = fuzz.membership.gaussmf(x_rgb, 19.896, 28.0547)
rmf1 = fuzz.membership.gaussmf(x_rgb,65.727,30.974)
rmf2 = fuzz.membership.gaussmf(x_rgb, 120.47, 36.400)
rmf3 = fuzz.membership.gaussmf(x_rgb, 185.43, 44.001)
#green
gmf0 = fuzz.membership.gaussmf(x_rgb, 1.414, 38.702)
gmf1 = fuzz.membership.gaussmf(x_rgb,64.97, 31.72)
gmf2 = fuzz.membership.gaussmf(x_rgb,129.39, 37.166)
gmf3 = fuzz.membership.gaussmf(x_rgb, 228.35, 61.241)
#blue
bmf0 = fuzz.membership.gaussmf(x_rgb,-8.30, 36.455)
bmf1 = fuzz.membership.gaussmf(x_rgb,52.79, 27.80)
bmf2 = fuzz.membership.gaussmf(x_rgb,140.755, 47.90)
bmf3 = fuzz.membership.gaussmf(x_rgb, 237.467, 62.499)
#entropy
emf0 = fuzz.membership.gaussmf(x_entropy, 1.33,0.778)
emf1 = fuzz.membership.gaussmf(x_entropy, 2.79, 0.503)
emf2 = fuzz.membership.gaussmf(x_entropy, 3.727,0.495)
emf3 = fuzz.membership.gaussmf(x_entropy, 4.715, 0.493)

# Visualize these universes and membership functions
fig3, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(4, 6))

ax0.plot(x_rgb, rmf0, 'r', linewidth=1.5, label='Low')
ax0.plot(x_rgb, rmf1, 'b', linewidth=1.5, label='Medium Low')
ax0.plot(x_rgb, rmf2, 'g', linewidth=1.5, label='Medium High')
ax0.plot(x_rgb, rmf3,'m', linewidth=1.5, label='High')
ax0.set_title('Red')
ax0.legend()

ax1.plot(x_rgb, gmf0, 'r', linewidth=1.5)
ax1.plot(x_rgb, gmf1, 'b', linewidth=1.5)
ax1.plot(x_rgb, gmf2, 'g', linewidth=1.5)
ax1.plot(x_rgb, gmf3,'m', linewidth=1.5)
ax1.set_title('Green')
ax1.legend()

ax2.plot(x_rgb, bmf0, 'r', linewidth=1.5)
ax2.plot(x_rgb, bmf1, 'b', linewidth=1.5)
ax2.plot(x_rgb, bmf2, 'g', linewidth=1.5)
ax2.plot(x_rgb, bmf3,'m', linewidth=1.5)
ax2.set_title('Blue')
ax2.legend()

ax3.plot(x_entropy, emf0, 'r', linewidth=1.5)
ax3.plot(x_entropy, emf1, 'b', linewidth=1.5)
ax3.plot(x_entropy, emf2, 'g', linewidth=1.5)
ax3.plot(x_entropy, emf3,'m', linewidth=1.5)
ax3.set_title('Entropy')
ax3.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()