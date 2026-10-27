import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.decomposition import PCA
import os

plt.rcParams['font.size'] = 20

# 1. Compute Top Principal Components
def get_top_pcs(features, num_pcs=3):
    pca = PCA(n_components=num_pcs)
    pca.fit(features)
    max_pc = np.diag(np.full(num_pcs, 0.1))
    top_pcs = pca.inverse_transform(max_pc)
    return top_pcs, pca.explained_variance_ratio_


# 2. Optimize Input for a Given PC
def optimize_for_pc(model, initial_input, static_input, pc_vector, device, num_steps=4000, lr=0.01):
    model.eval()

    optimized_input = initial_input.clone().detach().to(device).requires_grad_(True)
    static_input = static_input.to(device).detach()
    optimizer = optim.Adam([optimized_input], lr=lr)

    loss_history = []
    max_change_per_step = []
    first_step_tendency = None
    prev_input = optimized_input.detach().clone()

    for step in range(num_steps):
        optimizer.zero_grad()
        full_input = torch.cat([optimized_input, static_input], dim=0)
        output = model(full_input.unsqueeze(0))
        target = torch.tensor(pc_vector, dtype=torch.float32, device=device)
        loss = torch.sum((output - target) ** 2)
        loss.backward()

        if step == 0:
            first_step_tendency = optimized_input.grad.detach().cpu().squeeze().numpy()

        optimizer.step()

        with torch.no_grad():
            delta = (optimized_input - prev_input).abs()
            max_change = delta.max().item()
            max_change_per_step.append(max_change)
            prev_input = optimized_input.detach().clone()

        loss_history.append(loss.item())

        if step % 1000 == 0 or step == num_steps - 1:
            print(f"Step {step}: Loss = {loss.item():.6f} | Max Change: {max_change:.6f}")

    return optimized_input.detach(), loss_history, first_step_tendency, max_change_per_step


# 3. Run PCA and Optimize for Top PCs

plt.rcParams['font.size'] = 20

def run_backward_optimization(model, X_test, static_input_tensor, test_dataset, device, num_top_pcs=3, num_steps=4000, lr=0.01, output_dir="figures"):
    # 1. Compute top principal components
    top_pcs, explained_ratio = get_top_pcs(X_test, num_pcs=num_top_pcs)
    print("Top PCs shape:", top_pcs.shape)
    print("Explained Variance Ratio:", explained_ratio)

    optimized_inputs = []
    loss_curves = []
    tendencies = []
    max_change_curves = []

    for i in range(num_top_pcs):
        print(f"\nOptimizing input for PC {i+1}...")

        # Select initial input from the test dataset
        initial_input = torch.tensor(
            test_dataset.data.isel(pairs=0).values[0,0],
            device=device,
            dtype=torch.float32
        ).unsqueeze(0)

        # Optimize input for principal component direction
        optimized_dynamic, loss_history, tendency, max_change_curve = optimize_for_pc(
            model=model,
            initial_input=initial_input,
            static_input=static_input_tensor.squeeze(),
            pc_vector=top_pcs[i],
            device=device,
            num_steps=num_steps,
            lr=lr
        )

        # Store results
        full_input = torch.cat([optimized_dynamic, static_input_tensor.squeeze().to(device)], dim=0)
        optimized_inputs.append(full_input)
        loss_curves.append(loss_history)
        tendencies.append(tendency)
        max_change_curves.append(max_change_curve)

    optimized_inputs = torch.stack(optimized_inputs)  # Shape: [3, 5, 192, 288]

    # 2. Plot results
    os.makedirs(output_dir, exist_ok=True)
    plot_loss_curves(loss_curves, save_path=f"{output_dir}/loss_curves.png")
    plot_tendencies(tendencies, save_path=f"{output_dir}/tendencies.png")
    plot_optimized_inputs(optimized_inputs, save_path=f"{output_dir}/optimized_inputs.png")
    plot_difference_from_initial(optimized_inputs, initial_input, save_path=f"{output_dir}/difference_from_initial.png")

    return optimized_inputs, tendencies, explained_ratio

def plot_loss_curves(loss_curves, save_path=None):
    plt.figure(figsize=(10, 5))
    for i, loss_curve in enumerate(loss_curves):
        plt.plot(loss_curve, label=f"PC {i+1}")
    plt.title("Optimization Loss Curve for Each Principal Component")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tendencies(tendencies, save_path=None):
    tendencies_np = np.stack(tendencies, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    pc_titles = [f'First-step tendency for PC{i+1}' for i in range(3)]

    for i, ax in enumerate(axes.flat):
        tendency_plot = ax.imshow(tendencies_np[i], transform=ccrs.PlateCarree(),
                                  extent=[-180, 180, -90, 90], cmap='BrBG', origin='upper',
                                  vmin=-2e-5, vmax=2e-5)
        ax.set_title(pc_titles[i], fontsize=10, pad=4)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4, color="gray", alpha=0.5)
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        gl.top_labels = gl.right_labels = False

    plt.subplots_adjust(left=0.05, right=0.86, top=0.92, bottom=0.05, wspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.6])
    cbar = plt.colorbar(tendency_plot, cax=cbar_ax, orientation='vertical')
    cbar.set_label('First-Step Tendency (∇Loss)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimized_inputs(optimized_inputs, save_path=None):
    optimized_inputs_np = optimized_inputs[:, 0, :, :].squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    pc_titles = [f'Optimal inputs for PC{i+1}' for i in range(3)]

    for i, ax in enumerate(axes.flat):
        input_plot = ax.imshow(optimized_inputs_np[i], transform=ccrs.PlateCarree(),
                               extent=[-180, 180, -90, 90], cmap='viridis', origin='upper', vmin=-3, vmax=3)
        ax.set_title(pc_titles[i], fontsize=10, pad=4)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4, color="gray", alpha=0.5)
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        gl.top_labels = gl.right_labels = False

    plt.subplots_adjust(left=0.05, right=0.86, top=0.92, bottom=0.05, wspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.6])
    cbar = plt.colorbar(input_plot, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized TGCLDLLWP', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_from_initial(optimized_inputs, initial_input, save_path=None):
    optimized_inputs_np = optimized_inputs[:, 0, :, :].squeeze().cpu().numpy()
    initial_input_np = initial_input.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
    pc_titles = [f'PC{i+1} - initial input' for i in range(3)]

    for i, ax in enumerate(axes.flat):
        input_plot = ax.imshow(optimized_inputs_np[i] - initial_input_np, transform=ccrs.PlateCarree(),
                               extent=[-180, 180, -90, 90], cmap='BrBG', origin='upper', vmin=-0.5, vmax=0.5)
        ax.set_title(pc_titles[i], fontsize=10, pad=4)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4, color="gray", alpha=0.5)
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        gl.top_labels = gl.right_labels = False

    plt.subplots_adjust(left=0.05, right=0.86, top=0.92, bottom=0.05, wspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.6])
    cbar = plt.colorbar(input_plot, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized TGCLDLLWP', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

