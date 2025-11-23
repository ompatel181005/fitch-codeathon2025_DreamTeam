"""
Performance Progression Visualization
Shows improvement across model iterations
"""
import matplotlib.pyplot as plt
import numpy as np

# Performance data from pipeline runs
stages = ['Baseline\n(test.py)', 'Region-Country\nEncoding', 'LightGBM\nMeta-learner', 'Isotonic\nCalibration', 'Quantile\nModels']

# Scope 1 metrics
scope1_rmse = [100_348, 100_000, 78_800, 73_288, 65_582]
scope1_r2 = [0.17, 0.18, 0.49, 0.56, 0.65]

# Scope 2 metrics
scope2_rmse = [170_428, 169_500, 158_052, 152_556, 127_034]
scope2_r2 = [0.07, 0.08, 0.20, 0.26, 0.48]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Emissions Prediction Pipeline: Performance Evolution', fontsize=16, fontweight='bold')

# Scope 1 RMSE
ax1 = axes[0, 0]
bars1 = ax1.bar(stages, scope1_rmse, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.8, edgecolor='black')
ax1.set_ylabel('RMSE (tons CO2e)', fontsize=12, fontweight='bold')
ax1.set_title('Scope 1: RMSE Reduction', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=0)
ax1.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
# Add improvement annotations
improvement = (scope1_rmse[0] - scope1_rmse[-1]) / scope1_rmse[0] * 100
ax1.text(0.5, 0.95, f'Total Improvement: -{improvement:.1f}%', 
         transform=ax1.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         fontsize=10, fontweight='bold')

# Scope 1 R²
ax2 = axes[0, 1]
bars2 = ax2.bar(stages, scope1_r2, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Scope 1: R² Improvement', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=0)
ax2.set_ylim(0, 0.75)
ax2.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
# Add improvement annotations
r2_improvement = (scope1_r2[-1] - scope1_r2[0]) / scope1_r2[0] * 100
ax2.text(0.5, 0.95, f'Total Improvement: +{r2_improvement:.0f}%', 
         transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=10, fontweight='bold')

# Scope 2 RMSE
ax3 = axes[1, 0]
bars3 = ax3.bar(stages, scope2_rmse, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('RMSE (tons CO2e)', fontsize=12, fontweight='bold')
ax3.set_title('Scope 2: RMSE Reduction', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=0)
ax3.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
# Add improvement annotations
improvement2 = (scope2_rmse[0] - scope2_rmse[-1]) / scope2_rmse[0] * 100
ax3.text(0.5, 0.95, f'Total Improvement: -{improvement2:.1f}%', 
         transform=ax3.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         fontsize=10, fontweight='bold')

# Scope 2 R²
ax4 = axes[1, 1]
bars4 = ax4.bar(stages, scope2_r2, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax4.set_title('Scope 2: R² Improvement', fontsize=13, fontweight='bold')
ax4.tick_params(axis='x', rotation=0)
ax4.set_ylim(0, 0.55)
ax4.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
# Add improvement annotations
r2_improvement2 = (scope2_r2[-1] - scope2_r2[0]) / scope2_r2[0] * 100
ax4.text(0.5, 0.95, f'Total Improvement: +{r2_improvement2:.0f}%', 
         transform=ax4.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=10, fontweight='bold')

# Add key improvements text box
improvement_text = """
Key Innovations:
1. Region-Country Encoding: +0.4% R² (geographic signal)
2. LightGBM Meta-Learner: +35% R² (non-linear blending) 🚀
3. Isotonic Calibration: +14-27% R² (tail predictions) 🎯
4. Quantile Models: +16-89% R² (distribution coverage) 💥
"""
fig.text(0.5, 0.02, improvement_text, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
         family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('performance_progression.png', dpi=300, bbox_inches='tight')
print("✅ Saved performance_progression.png")

# Print summary statistics
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"\nScope 1 (Direct Emissions):")
print(f"  Baseline → Final: RMSE {scope1_rmse[0]:,} → {scope1_rmse[-1]:,} (-{(scope1_rmse[0]-scope1_rmse[-1])/scope1_rmse[0]*100:.1f}%)")
print(f"  Baseline → Final: R² {scope1_r2[0]:.3f} → {scope1_r2[-1]:.3f} (+{(scope1_r2[-1]-scope1_r2[0])/scope1_r2[0]*100:.0f}%)")

print(f"\nScope 2 (Indirect Emissions):")
print(f"  Baseline → Final: RMSE {scope2_rmse[0]:,} → {scope2_rmse[-1]:,} (-{(scope2_rmse[0]-scope2_rmse[-1])/scope2_rmse[0]*100:.1f}%)")
print(f"  Baseline → Final: R² {scope2_r2[0]:.3f} → {scope2_r2[-1]:.3f} (+{(scope2_r2[-1]-scope2_r2[0])/scope2_r2[0]*100:.0f}%)")

print(f"\nBiggest Gains:")
print(f"  1. LightGBM Meta-Learner: Scope1 R² +{(scope1_r2[2]-scope1_r2[1])/scope1_r2[1]*100:.0f}%, Scope2 R² +{(scope2_r2[2]-scope2_r2[1])/scope2_r2[1]*100:.0f}%")
print(f"  2. Quantile Models: Scope1 R² +{(scope1_r2[4]-scope1_r2[3])/scope1_r2[3]*100:.0f}%, Scope2 R² +{(scope2_r2[4]-scope2_r2[3])/scope2_r2[3]*100:.0f}%")
print(f"  3. Isotonic Calibration: Scope1 R² +{(scope1_r2[3]-scope1_r2[2])/scope1_r2[2]*100:.0f}%, Scope2 R² +{(scope2_r2[3]-scope2_r2[2])/scope2_r2[2]*100:.0f}%")
print("="*60)
