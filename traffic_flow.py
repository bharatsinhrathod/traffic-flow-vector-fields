"""
Traffic Flow Analysis using Vector Fields
SEM-4 Vector Calculus Practical Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

# ─────────────────────────────────────────────
# 1. GRID SETUP
# ─────────────────────────────────────────────
X_RANGE = (0, 10)
Y_RANGE = (0, 10)
N = 30   # grid resolution

x = np.linspace(*X_RANGE, N)
y = np.linspace(*Y_RANGE, N)
X, Y = np.meshgrid(x, y)

# ─────────────────────────────────────────────
# 2. VECTOR FIELD DEFINITION
#    F(x,y) = (Fx, Fy) — traffic velocity field
#    Obstacles create divergence/convergence
# ─────────────────────────────────────────────
# Base rightward flow
Fx = np.ones_like(X) * 1.5
Fy = np.zeros_like(Y)

# Add lane-like sinusoidal variation
Fx += 0.4 * np.sin(Y * np.pi / 5)
Fy += 0.2 * np.cos(X * np.pi / 5)

# Obstacles (road blocks / intersections)
obstacles = [
    (3.5, 4.5, 0.8),   # (cx, cy, radius)
    (6.5, 3.0, 0.7),
    (5.0, 7.0, 0.6),
    (2.0, 6.5, 0.5),
    (7.8, 6.0, 0.6),
]

for (ox, oy, r) in obstacles:
    dist2 = (X - ox)**2 + (Y - oy)**2
    dist  = np.sqrt(dist2) + 1e-9
    # Potential flow deflection around obstacle (irrotational)
    Fx += r**2 * (X - ox) / dist2**1.5 * 2.5
    Fy += r**2 * (Y - oy) / dist2**1.5 * 2.5
    # Suppress flow inside obstacle
    mask = dist < r * 0.9
    Fx[mask] = 0
    Fy[mask] = 0

# Speed magnitude
Speed = np.sqrt(Fx**2 + Fy**2)

# ─────────────────────────────────────────────
# 3. DIVERGENCE  ∇·F = ∂Fx/∂x + ∂Fy/∂y
#    Negative divergence → Convergence (Congestion)
#    Positive divergence → Divergence (Free flow)
# ─────────────────────────────────────────────
dFx_dx = np.gradient(Fx, x, axis=1)
dFy_dy = np.gradient(Fy, y, axis=0)
Divergence = dFx_dx + dFy_dy

# ─────────────────────────────────────────────
# 4. CURL  ∇×F = ∂Fy/∂x − ∂Fx/∂y
#    Indicates rotational flow (roundabouts)
# ─────────────────────────────────────────────
dFy_dx = np.gradient(Fy, x, axis=1)
dFx_dy = np.gradient(Fx, y, axis=0)
Curl = dFy_dx - dFx_dy


os.makedirs("/home/claude/graphs", exist_ok=True)

# ─────────────────────────────────────────────
# GRAPH 1 — Vector Field with Streamlines
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
speed_norm = Speed / Speed.max()
strm = ax.streamplot(X, Y, Fx, Fy, color=speed_norm,
                     cmap='RdYlGn', linewidth=1.4,
                     density=1.6, arrowsize=1.3)
cbar = fig.colorbar(strm.lines, ax=ax, shrink=0.85)
cbar.set_label("Normalised speed (0 = congested, 1 = free)", fontsize=10)

for (ox, oy, r) in obstacles:
    circle = plt.Circle((ox, oy), r, color='#534AB7', alpha=0.55, zorder=5)
    ax.add_patch(circle)
    ax.text(ox, oy, 'OBS', ha='center', va='center',
            color='white', fontsize=7, fontweight='bold', zorder=6)

ax.set_xlim(*X_RANGE); ax.set_ylim(*Y_RANGE)
ax.set_xlabel("X (road grid)", fontsize=11)
ax.set_ylabel("Y (road grid)", fontsize=11)
ax.set_title("Fig 1 — Traffic Flow Vector Field (Streamlines)", fontsize=13, pad=12)
ax.set_facecolor('#F7F7F5')
fig.tight_layout()
fig.savefig("fig1_vector_field.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ─────────────────────────────────────────────
# GRAPH 2 — Divergence Map (Congestion Detection)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
div_lim = np.percentile(np.abs(Divergence), 97)
cf = ax.contourf(X, Y, Divergence, levels=40,
                 cmap='RdBu_r', vmin=-div_lim, vmax=div_lim)
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label("Divergence ∇·F", fontsize=10)

# Mark congestion hotspots (strong negative divergence)
cong_mask = Divergence < -div_lim * 0.55
if cong_mask.any():
    ax.contour(X, Y, Divergence, levels=[-div_lim * 0.55],
               colors='darkred', linewidths=1.8, linestyles='--')

for (ox, oy, r) in obstacles:
    circle = plt.Circle((ox, oy), r, color='#534AB7', alpha=0.4, zorder=5)
    ax.add_patch(circle)

# Quiver overlay (thin arrows)
step = 3
ax.quiver(X[::step, ::step], Y[::step, ::step],
          Fx[::step, ::step], Fy[::step, ::step],
          alpha=0.35, scale=35, width=0.003, color='#2C2C2A')

ax.set_xlim(*X_RANGE); ax.set_ylim(*Y_RANGE)
ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
ax.set_title("Fig 2 — Divergence Map  (Red = Congestion, Blue = Dispersion)", fontsize=13, pad=12)
red_p   = mpatches.Patch(color='#C0392B', label='Convergence (congestion)')
blue_p  = mpatches.Patch(color='#2980B9', label='Divergence (free flow)')
obs_p   = mpatches.Patch(color='#534AB7', alpha=0.6, label='Obstacle')
dash_p  = mpatches.Patch(color='darkred', label='Congestion boundary')
ax.legend(handles=[red_p, blue_p, obs_p, dash_p], loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig("fig2_vector_field.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ─────────────────────────────────────────────
# GRAPH 3 — Speed Heatmap
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
hm = ax.pcolormesh(X, Y, Speed, cmap='RdYlGn', shading='auto')
cbar = fig.colorbar(hm, ax=ax, shrink=0.85)
cbar.set_label("Speed magnitude |F(x,y)|", fontsize=10)

step = 4
ax.quiver(X[::step, ::step], Y[::step, ::step],
          Fx[::step, ::step], Fy[::step, ::step],
          alpha=0.5, scale=40, width=0.003, color='black')

for (ox, oy, r) in obstacles:
    circle = plt.Circle((ox, oy), r, color='#534AB7', alpha=0.7, zorder=5)
    ax.add_patch(circle)
    ax.text(ox, oy, 'OBS', ha='center', va='center',
            color='white', fontsize=7, fontweight='bold', zorder=6)

ax.set_xlim(*X_RANGE); ax.set_ylim(*Y_RANGE)
ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
ax.set_title("Fig 3 — Speed Heatmap (Green = Fast, Red = Slow)", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig("fig3_vector_field.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3")

# ─────────────────────────────────────────────
# GRAPH 4 — Curl Map (Rotational Flow)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
curl_lim = np.percentile(np.abs(Curl), 97)
cf2 = ax.contourf(X, Y, Curl, levels=40,
                  cmap='PuOr', vmin=-curl_lim, vmax=curl_lim)
cbar = fig.colorbar(cf2, ax=ax, shrink=0.85)
cbar.set_label("Curl ∇×F", fontsize=10)

for (ox, oy, r) in obstacles:
    circle = plt.Circle((ox, oy), r, color='#0F6E56', alpha=0.5, zorder=5)
    ax.add_patch(circle)

ax.set_xlim(*X_RANGE); ax.set_ylim(*Y_RANGE)
ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
ax.set_title("Fig 4 — Curl Map  (Orange = Clockwise, Purple = Anti-clockwise)", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig("fig4_vector_field.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ─────────────────────────────────────────────
# GRAPH 5 — Line Integral along a path
#    ∫_C F·dr  (work done by flow along route)
# ─────────────────────────────────────────────
t = np.linspace(0, 1, 500)
path_x = X_RANGE[0] + t * (X_RANGE[1] - X_RANGE[0])
path_y = 5 + 2 * np.sin(t * 2 * np.pi)      # sinusoidal road path

# Interpolate field at path points
from scipy.interpolate import RegularGridInterpolator
interp_Fx = RegularGridInterpolator((y, x), Fx, method='linear', bounds_error=False, fill_value=0)
interp_Fy = RegularGridInterpolator((y, x), Fy, method='linear', bounds_error=False, fill_value=0)

pts = np.column_stack([path_y, path_x])
Fx_path = interp_Fx(pts)
Fy_path = interp_Fy(pts)

dx_dt = np.gradient(path_x, t)
dy_dt = np.gradient(path_y, t)
integrand = Fx_path * dx_dt + Fy_path * dy_dt
cumulative = np.cumsum(integrand) * (t[1] - t[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: path on vector field
ax0 = axes[0]
ax0.streamplot(X, Y, Fx, Fy, color=Speed / Speed.max(),
               cmap='RdYlGn', linewidth=0.9, density=1.2, arrowsize=1.0)
ax0.plot(path_x, path_y, 'b-', linewidth=2.5, label='Vehicle path C')
ax0.plot(path_x[0], path_y[0], 'go', markersize=9, label='Start')
ax0.plot(path_x[-1], path_y[-1], 'rs', markersize=9, label='End')
for (ox, oy, r) in obstacles:
    ax0.add_patch(plt.Circle((ox, oy), r, color='#534AB7', alpha=0.5))
ax0.set_xlim(*X_RANGE); ax0.set_ylim(*Y_RANGE)
ax0.set_xlabel("X"); ax0.set_ylabel("Y")
ax0.set_title("Vehicle path on vector field")
ax0.legend(fontsize=8)

# Right: cumulative line integral
ax1 = axes[1]
ax1.plot(path_x, cumulative, color='#185FA5', linewidth=2)
ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax1.fill_between(path_x, cumulative, 0,
                 where=cumulative > 0, alpha=0.2, color='green', label='Aided flow')
ax1.fill_between(path_x, cumulative, 0,
                 where=cumulative < 0, alpha=0.2, color='red', label='Resisted flow')
ax1.set_xlabel("X position along path")
ax1.set_ylabel("Cumulative ∫F·dr")
ax1.set_title(f"Line integral along path  (Total = {cumulative[-1]:.2f})")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

fig.suptitle("Fig 5 — Line Integral of Traffic Flow Along a Vehicle Path", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("fig5_vector_field.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ─────────────────────────────────────────────
# PRINT NUMERICAL RESULTS
# ─────────────────────────────────────────────
print("\n========= ANALYSIS RESULTS =========")
print(f"Average speed          : {Speed.mean():.3f}")
print(f"Max speed              : {Speed.max():.3f}")
print(f"Min speed              : {Speed.min():.3f}")
print(f"Max divergence (∇·F)   : {Divergence.max():.3f}")
print(f"Min divergence (∇·F)   : {Divergence.min():.3f}")
print(f"Congestion cells (∇·F<0): {(Divergence < 0).sum()} / {Divergence.size}")
print(f"Max curl (∇×F)         : {Curl.max():.3f}")
print(f"Line integral ∫F·dr    : {cumulative[-1]:.3f}")
print("=====================================")
print("\nAll 5 graphs saved to /home/claude/graphs/")
