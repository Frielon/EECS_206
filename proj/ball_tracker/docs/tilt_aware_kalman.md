# Tilt-Aware Kalman Filter for Ball-on-Plate Tracking

## 1. Motivation

The current ball tracker (`tracker/kalman_filter.py`) uses a **constant-velocity** motion
model. On a tilting plate this is structurally wrong: the ball is constantly accelerating
under projected gravity. The filter compensates by absorbing gravity-induced motion into
the process-noise term `Q`, which forces an unfortunate trade-off:

| `KF_PROCESS_NOISE` | Behavior                                     |
|--------------------|----------------------------------------------|
| High               | Tracks fast motion, but estimates are noisy  |
| Low                | Smooth estimates, but lag real motion        |

For closed-loop control, both noise and lag are destabilizing — noise in the velocity
channel becomes derivative-action chatter, and lag introduces an effective dead time
that the controller must back off to tolerate.

The plate's tilt is **known** at every instant (it is the controller's own command).
Folding it into the KF as a deterministic input lets the model account for the
*expected* acceleration directly, so `Q` only needs to absorb the much smaller
*unmodeled* acceleration (slip, rolling resistance, sin-approximation error, plate
angle backlash). This shrinks both noise and lag at the same time.

## 2. Physics

For a uniform solid sphere rolling without slip on a plate tilted by angles
`(θ_x, θ_y)` (rotations about the table x- and y-axes), the in-plane acceleration of
the ball's center, expressed in the table frame, is:

```
a_x = +α · g · sin(θ_y)
a_y = -α · g · sin(θ_x)
```

where:

- `g ≈ 9810 mm/s²` (gravitational acceleration in the units we already use).
- `α = 1 / (1 + I / (m·r²))`. For a uniform solid sphere, `I = (2/5) m r²`, giving
  `α = 5/7 ≈ 0.714`. Use `α = 1` for pure sliding (e.g., a low-friction puck).

### Sign convention

This must match the physical setup. The defaults in `config.py` use:

```
Table x-axis points right, y-axis points up (looking from camera).
```

With right-hand-rule rotations:

- `θ_y > 0` ⇒ the +x edge of the plate dips down ⇒ ball accelerates toward +x.
- `θ_x > 0` ⇒ the +y edge of the plate dips down ⇒ ball accelerates toward -y.

**Sanity check before trusting the controller:** tilt the plate by hand to a known
direction, command zero motor output, and verify that the predicted velocity arrow in
the debug overlay points the same way the ball actually rolls. If it points the wrong
way, flip the corresponding sign in `tilt_to_accel`.

## 3. Kalman Formulation

State: `x = [x, y, vx, vy]ᵀ`, with positions in mm and velocities in mm/s.
Measurement: `z = [x, y]ᵀ` (unchanged).

Discrete-time model with control input `u = [a_x, a_y]ᵀ` and time step `Δt`:

```
x_{k+1} = F · x_k + B · u_k + w_k       w_k ~ N(0, Q)
z_k     = H · x_k + v_k                 v_k ~ N(0, R)
```

with

```
        ┌1  0  Δt  0 ┐         ┌Δt²/2     0  ┐
   F =  │0  1   0  Δt│    B =  │   0   Δt²/2 │
        │0  0   1   0│         │  Δt      0  │
        └0  0   0   1┘         └   0     Δt  ┘
```

`H` and `R` are unchanged. `Q` keeps the same continuous-white-noise-acceleration
structure as before:

```
        ┌Δt⁴/4    0    Δt³/2    0   ┐
   Q =  │  0    Δt⁴/4    0    Δt³/2 │ · σ_a²
        │Δt³/2    0     Δt²    0    │
        └  0    Δt³/2    0     Δt²  ┘
```

but now `σ_a` is interpreted as the **unmodeled** acceleration noise, not the total
acceleration. Once the gravity component is in the model, `KF_PROCESS_NOISE` typically
needs to drop by ~5–10× from its old value (e.g., 5.0 → 0.5–1.0). Re-tune after the
change.

### Predict / Update equations

```
predict:
    x ← F·x + B·u
    P ← F·P·Fᵀ + Q

update (unchanged):
    y ← z - H·x
    S ← H·P·Hᵀ + R
    K ← P·Hᵀ·S⁻¹
    x ← x + K·y
    P ← (I - K·H)·P
```

## 4. API Changes

### `tracker/kalman_filter.py`

- New module-level helper:
  ```python
  tilt_to_accel(theta_x, theta_y, g=GRAVITY_MM_S2, alpha=ROLLING_FACTOR) -> (a_x, a_y)
  ```
  Converts plate tilt angles (radians) to table-frame acceleration (mm/s²).
- New method on `BallKalmanFilter`:
  ```python
  set_control(a_x, a_y)   # store latest u; applied by predict()
  ```
- `predict()` adds `B · u` to the state.
- The `B` matrix is rebuilt whenever `dt` changes, alongside `F` and `Q`.
- Default `u` is `[0, 0]`, so a filter that never receives a control input behaves
  exactly like the old constant-velocity one (no regression).

### `tracker/pipeline.py`

- New attribute `self.plate_angles = (0.0, 0.0)`.
- New method `set_plate_angles(theta_x, theta_y)` — call this whenever the controller
  updates the commanded tilt.
- Inside `process_frame`, before `kf.predict()`, the pipeline computes `(a_x, a_y)`
  via `tilt_to_accel` and pushes it to the filter via `set_control`.

### `config.py`

- New constants:
  ```python
  GRAVITY_MM_S2  = 9810.0
  ROLLING_FACTOR = 5.0 / 7.0   # uniform solid sphere; use 1.0 for sliding
  ```

## 5. Integration with the Control Loop

```python
def controller_callback(result):
    if not result['ball_found']:
        return

    x, y   = result['position']
    vx, vy = result['velocity']

    # PD example, in radians
    Kp, Kd = 0.01, 0.005
    theta_x = -Kp * y - Kd * vy   # tilt about x-axis controls y motion
    theta_y =  Kp * x + Kd * vx   # tilt about y-axis controls x motion

    pipeline.set_plate_angles(theta_x, theta_y)   # feed back into KF
    send_to_motors(theta_x, theta_y)
```

Important: feed the **commanded** angle to the filter before sending it to the motors,
or — better — feed back whatever angle the motors actually achieve if you have
position feedback on the tilt actuators. Stale angles produce stale predictions.

## 6. Validation

Three sanity checks, in order of cost:

1. **Direction check.** Hold the plate at a fixed tilt by hand. With the controller
   disabled, call `set_plate_angles(theta_x, theta_y)` from a small script. Release
   the ball at rest. The KF velocity arrow should point the same way the ball rolls.
   If reversed, flip a sign in `tilt_to_accel`.
2. **Magnitude check.** Tilt the plate to a measured angle (e.g., 5°). After ~0.3 s the
   measured velocity from the KF should match `α · g · sin(θ) · t` to within ~10 %.
   Larger discrepancy ⇒ slipping (try `α = 1`) or wrong angle calibration.
3. **Closed-loop comparison.** With the plate flat, push the ball by hand. Log
   `result['velocity']` versus the numerical derivative of `result['position']`.
   The new filter should be smoother *and* less laggy than the old one at the same
   `KF_PROCESS_NOISE`.

## 7. Limitations and Future Work

- **Commanded ≠ achieved tilt.** If the motors have backlash, low bandwidth, or
  saturate, the model diverges from reality. Mitigation: use measured plate angle
  (e.g., from servo encoders) instead of the command, or add a first-order plate
  dynamics model.
- **No friction term.** Coulomb friction creates a small dead band at low tilt; this
  shows up as a constant drift in the velocity estimate near rest. A simple sign-of-
  velocity friction model can be added later if it matters.
- **Rolling factor assumes rolling without slipping.** A smooth ball on a smooth plate
  can slip, especially during fast direction changes; the effective `α` then sits
  somewhere between `5/7` and `1`. Treat `ROLLING_FACTOR` as a tunable.
- **Gravity-aligned table frame.** If the rig is mounted at a tilt, add a static
  offset to `(θ_x, θ_y)` so a "zero command" still includes the resting bias.
- **Improves the temporal model only.** The measurement still has a parallax bias
  because the ball's center sits at height `r` above the marker plane. That is the
  scope of improvement #2 (parallax / ball-height correction).
