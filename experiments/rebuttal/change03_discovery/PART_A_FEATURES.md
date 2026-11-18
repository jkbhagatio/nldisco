# Part A: prespecified behavioral features

Written and committed before session B's data existed and before any model was
trained on or inspected for session B. The executable definitions are in
`features.py`. No definition refers to a latent of any method: every feature is a
condition on the recorded behavior variables, chosen to cover each variable family
available for the Aeon recording (position, speed, acceleration, heading, movement
relative to the patch, body area, wheel speed, food delivery).

All variables are read at the last 20 ms bin of each 0.4 s neural window. A window
enters a feature's comparison only if its whole 20-bin span has valid samples of
the variables involved. Thresholds are fixed physical values, except where a
quantile of the discovery part's behavior is stated; quantiles and the modal
position never use the held-out part.

| # | Feature | Condition | Compared within |
|---|---------|-----------|-----------------|
| 1 | near the patch | patch distance <= 100 px | windows with valid tracking |
| 2 | at the most occupied location | within 100 px of the most occupied 25-px cell of the discovery part | windows with valid tracking |
| 3 | elsewhere in the arena | neither 1 nor 2 | windows with valid tracking |
| 4 | stationary | speed < 5 px/s | windows with valid tracking |
| 5 | moving | speed >= 20 px/s | windows with valid tracking |
| 6 | fast locomotion | speed >= 100 px/s | windows with valid tracking |
| 7 | accelerating | acceleration >= 200 px/s^2 | windows with valid tracking |
| 8 | decelerating | acceleration <= -200 px/s^2 | windows with valid tracking |
| 9 | moving right | movement direction within 45 deg of +x | moving windows (speed >= 20 px/s) |
| 10 | moving down | movement direction within 45 deg of +y (image coordinates) | moving windows |
| 11 | moving left | movement direction within 45 deg of -x | moving windows |
| 12 | moving up | movement direction within 45 deg of -y | moving windows |
| 13 | moving toward the patch | movement direction within 45 deg of the direction to the patch | moving windows |
| 14 | moving away from the patch | movement direction at least 135 deg from the direction to the patch | moving windows |
| 15 | small body area | area <= lowest-quintile edge of the discovery part | windows with valid tracking |
| 16 | large body area | area >= highest-quintile edge of the discovery part | windows with valid tracking |
| 17 | wheel running | wheel speed > 0.05 turns/s | windows with valid wheel samples |
| 18 | fast wheel running | wheel speed >= median wheel speed of wheel-running windows in the discovery part | wheel-running windows |
| 19 | wheel-bout onset | 0 to 0.5 s after a wheel bout starts | windows with valid wheel samples |
| 20 | wheel-bout offset | 0 to 0.5 s after a wheel bout ends | windows with valid wheel samples |
| 21 | at the patch, wheel idle | patch distance <= 100 px and wheel speed <= 0.02 turns/s | windows with valid tracking and wheel samples |
| 22 | after a pellet trigger | 0 to 2 s after a pellet trigger | all windows |
| 23 | before a pellet trigger | 0 to 1 s before a pellet trigger | all windows |

A wheel bout is wheel speed above 0.05 turns/s, with pauses shorter than 0.2 s
filled and bouts shorter than 0.4 s removed. Movement direction is the direction
of centroid velocity and is defined only above 20 px/s.

A feature is scored for a method only if its condition holds in at least 500
windows and fails in at least 500 windows of both the discovery and the held-out
part; otherwise it is reported as not scorable for every method.
