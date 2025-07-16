# NPC-l2race
## Goal: Implementing a basic neuromorphic predictive controller for a racing car that adapts to unknown racing car track environments.


MLP is a type of NN architecture that is foundational in DL. It is simple, classic, and powerful when used right
-Data flows forward from input to output (Feedforward).
-Network learns by adjusting Weights using BackProp and Gradient descent
-Activation Functions(ReLu, sigmoid or tanh) introduces non-linearity, thus allowing to learn complex patters
Note:
- Not ideal for sequential/spatial data (images (CNN), time series, or language(use RNN))

MLP is serving as a lightweight policy Net or correction module that will support our Pure Pursuit baseline control.

Logging to track:
- State, actions, corrections
- Rewards (e.g., lateral deviation or lap progress)
- Online learning loss

Model saving
 - save brain bt runs and dump to disk with torch.save.

Feature                 Purpose
log.csv                 Debug what the bot is actually doing
correction_net.pth      Save the model so you can reload it next run
Logging class           Append every frame or batch safely



+-------------------------+ </br>
|   Server Track Engine   |  ← (Blackbox Car Dynamics) </br>
+-----------+-------------+ </br>
            | 
            v </br>
     Full State (x, y, vx, vy, ax, ay, heading, etc.) </br>
            | </br>
            v </br>
+-------------------------+
|     CarController.py    |
+-------------------------+
|                         |
| 1. _get_state_features()|  ← Builds 10D vector: sₜ
|                         |
| 2. _get_lookahead_target()  ← Finds L meters ahead on centerline
|                         |
| 3. _compute_angle_to_target()  ← α = angle to lookahead point
|                         |
| 4. _pure_pursuit_steer(α)     ← δ₀ = base steering
|                         |
| 5. _curvature_based_throttle() ← τ₀ = base throttle
|                         |
| 6. CorrectionNet.predict(sₜ) ← Δδ, Δτ
|                         |
| 7. Combine: δ = δ₀ + Δδ ; τ = τ₀ + Δτ
|                         |
| 8. Send CarCommand(δ, τ, 0) → server
|                         |
| 9. Log (sₜ, δ, τ, Δδ, Δτ, centerline error, loss)
|                         |
| 10. _online_update() → trains CorrectionNet
+-------------------------+
            |
            v
     CorrectionNet.pth  ← learned residuals saved

2. MATHEMATICAL FORMULAS & LOGIC

   1. Car State Feature Vector (sₜ)
    sₜ = [x, y, vx, vy, ax, ay, dist_to_centerline, angle_to_centerline, curvature, surface_type]

    Used as input to:
    - CorrectionNet (MLP)
    - Residual learning
    - Logging / training

    2. Lookahead Target (Pure Pursuit Target)
    Given:
    - P_car = current position
    - Centerline = [p₀, p₁, ..., pₙ]
    - We find the closest segment, then walk ahead L meters to get the point P_L such that: </br>
            $$
       \sum_{jclosest} \|p_{i+1} - p_i\| \geq L
      $$
            </br>And interpolate if overshooting.

    



