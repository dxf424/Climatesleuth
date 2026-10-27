import torch
from captum.attr import IntegratedGradients
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
HERE = os.path.dirname(os.path.abspath(__file__))
lon_grid = np.load(os.path.join(HERE, "lon_grid.npy"))
lat_grid = np.load(os.path.join(HERE, "lat_grid.npy"))
plt.rcParams['font.size'] = 11

def make_euclidean_distance_loss(model):
    def euclidean_distance_loss(x1, x2):
        z_i = model(x1)
        z_j = model(x2)
        return F.pairwise_distance(z_i, z_j, p=2).mean().unsqueeze(0)
    return euclidean_distance_loss

def compute_integrated_gradients(
    model,
    test_loader,
    test_dataset,
    static_input_tensor,
    device,
    max_batches=None,
    n_steps=100,
    output_dir="figures"
):
    """
    Compute Integrated Gradients for a given model and test dataset.

    Args:
        model: Trained contrastive model.
        test_loader: DataLoader yielding batch indices.
        test_dataset: xarray dataset containing "data", "week", and "member".
        static_input_tensor: Tensor of static features, shape [1, C, H, W].
        device: CUDA or CPU device.
        max_batches: Optional limit to the number of batches processed.

    Returns:
        sorted_results: List of dicts containing integrated gradients and metadata.
    """
    model.to(device)
    model.eval()

    static_repeated = static_input_tensor.repeat(2, 1, 1, 1).to(device, non_blocking=True)

    results = []
    count = 0

    for batch_indices in test_loader:
        batch_indices = batch_indices[batch_indices < test_dataset.sizes["pairs"]]
        if len(batch_indices) == 0:
            continue

        week = test_dataset.week.isel(pairs=batch_indices).values[0]
        if isinstance(week, str) and (week.startswith("0002") or week.startswith("0003")):
            continue

        PPE_member = test_dataset.member.isel(pairs=batch_indices).values[0]

        data_batch = test_dataset["data"].isel(pairs=batch_indices).load()

        x1 = torch.from_numpy(data_batch.isel(member=0).values).float().unsqueeze(1).to(device)
        x2 = torch.from_numpy(data_batch.isel(member=1).values).float().unsqueeze(1).to(device)
        x1_combined = torch.cat([x1, static_repeated], dim=1)
        x2_combined = torch.cat([x2, static_repeated], dim=1)
        
        # 🔹 **Pass the wrapper function to IntegratedGradients**
        ig = IntegratedGradients(make_euclidean_distance_loss(model)) 

        attributions, delta = ig.attribute(
            (x1_combined[:1], x1_combined[1:]),
            baselines=(x1_combined[1:], x1_combined[1:]), 
            return_convergence_delta=True,
            n_steps = n_steps
        )

        # Split into ML and KK (custom logic—assumes index=1 for now)
        index = 1
        # Gradients
        gradient_ML = attributions[0].detach().cpu().numpy()
        gradient_KK = attributions[1].detach().cpu().numpy()
        mean_gradient_ML = gradient_ML.mean(axis=0)  # Mean over batch
        mean_gradient_KK = gradient_KK.mean(axis=0)

        # Inputs
        inputs_ML = x1_combined[:index].detach().cpu().numpy()
        inputs_KK = x1_combined[index:].detach().cpu().numpy()
        mean_inputs_ML = inputs_ML.mean(axis=0)
        mean_inputs_KK = inputs_KK.mean(axis=0)

        results.append({
            "week": week.item() if hasattr(week, "item") else week,
            "PPE_member": PPE_member.item() if hasattr(PPE_member, "item") else PPE_member,
            "mean_gradient_ML": mean_gradient_ML,
            "mean_gradient_KK": mean_gradient_KK,
            "mean_inputs_ML": mean_inputs_ML,
            "mean_inputs_KK": mean_inputs_KK,
        })

        count += 1
        if max_batches and count >= max_batches:
            break
        if count % 100 == 0:
            print(f"Processed {count} batches")

    sorted_results = sorted(results, key=lambda x: (x["week"], x["PPE_member"]))

    #plot 
    os.makedirs(output_dir, exist_ok=True)
    plot_mean_fields_and_abs_grads(sorted_results, save_path=f"{output_dir}/mean_inputs_and_gradients.png")
    return sorted_results

def compute_integrated_gradients_monthly_data(
    model,
    test_loader,
    test_dataset,
    static_input_tensor,
    device,
    max_batches=None,
    n_steps=100,
    output_dir="figures"
):
    """
    Compute Integrated Gradients for a given model and test dataset.

    Args:
        model: Trained contrastive model.
        test_loader: DataLoader yielding batch indices.
        test_dataset: xarray dataset containing "data", "month", and "member".
        static_input_tensor: Tensor of static features, shape [1, C, H, W].
        device: CUDA or CPU device.
        max_batches: Optional limit to the number of batches processed.

    Returns:
        sorted_results: List of dicts containing integrated gradients and metadata.
    """
    model.to(device)
    model.eval()

    static_repeated = static_input_tensor.repeat(2, 1, 1, 1).to(device, non_blocking=True)

    results = []
    count = 0

    for batch_indices in test_loader:
        batch_indices = batch_indices[batch_indices < test_dataset.sizes["pairs"]]
        if len(batch_indices) == 0:
            continue

        month = test_dataset.month.isel(pairs=batch_indices).values[0,0]

        PPE_member = test_dataset.member.isel(pairs=batch_indices).values[0]

        data_batch = test_dataset["data"].isel(pairs=batch_indices).load()

        x1 = torch.from_numpy(data_batch.isel(pair=0).values).float().to(device)
        x2 = torch.from_numpy(data_batch.isel(pair=1).values).float().to(device)
        x1_combined = torch.cat([x1, static_repeated], dim=1)
        x2_combined = torch.cat([x2, static_repeated], dim=1)
        
        # 🔹 **Pass the wrapper function to IntegratedGradients**
        ig = IntegratedGradients(make_euclidean_distance_loss(model)) 

        attributions, delta = ig.attribute(
            (x1_combined[:1], x1_combined[1:]),
            baselines=(x1_combined[1:], x1_combined[1:]), 
            return_convergence_delta=True,
            n_steps = n_steps
        )

        # Split into ML and KK (custom logic—assumes index=1 for now)
        index = 1
        # Gradients
        gradient_ML = attributions[0].detach().cpu().numpy()
        gradient_KK = attributions[1].detach().cpu().numpy()
        mean_gradient_ML = gradient_ML.mean(axis=0)  # Mean over batch
        mean_gradient_KK = gradient_KK.mean(axis=0)

        # Inputs
        inputs_ML = x1_combined[:index].detach().cpu().numpy()
        inputs_KK = x1_combined[index:].detach().cpu().numpy()
        mean_inputs_ML = inputs_ML.mean(axis=0)
        mean_inputs_KK = inputs_KK.mean(axis=0)

        results.append({
            "month": month.item() if hasattr(month, "item") else month,
            "PPE_member": PPE_member.item() if hasattr(PPE_member, "item") else PPE_member,
            "mean_gradient_ML": mean_gradient_ML,
            "mean_gradient_KK": mean_gradient_KK,
            "mean_inputs_ML": mean_inputs_ML,
            "mean_inputs_KK": mean_inputs_KK,
        })

        if month == '2002-11' or '0002-11':
            # 🔹 **Pass the wrapper function to IntegratedGradients**
            ig = IntegratedGradients(make_euclidean_distance_loss(model)) 

            attributions, delta = ig.attribute(
                (x2_combined[:1], x2_combined[1:]),
                baselines=(x2_combined[1:], x2_combined[1:]), 
                return_convergence_delta=True,
                n_steps = n_steps
            )

            # Split into ML and KK (custom logic—assumes index=1 for now)
            index = 1
            # Gradients
            gradient_ML = attributions[0].detach().cpu().numpy()
            gradient_KK = attributions[1].detach().cpu().numpy()
            mean_gradient_ML = gradient_ML.mean(axis=0)  # Mean over batch
            mean_gradient_KK = gradient_KK.mean(axis=0)

            # Inputs
            inputs_ML = x2_combined[:index].detach().cpu().numpy()
            inputs_KK = x2_combined[index:].detach().cpu().numpy()
            mean_inputs_ML = inputs_ML.mean(axis=0)
            mean_inputs_KK = inputs_KK.mean(axis=0)

            results.append({
                "month": month.item() if hasattr(month, "item") else month,
                "PPE_member": PPE_member.item() if hasattr(PPE_member, "item") else PPE_member,
                "mean_gradient_ML": mean_gradient_ML,
                "mean_gradient_KK": mean_gradient_KK,
                "mean_inputs_ML": mean_inputs_ML,
                "mean_inputs_KK": mean_inputs_KK,
            })
        count += 1
        if max_batches and count >= max_batches:
            break
        if count % 100 == 0:
            print(f"Processed {count} batches")

    sorted_results = sorted(results, key=lambda x: (x["month"], x["PPE_member"]))

    #plot 
    os.makedirs(output_dir, exist_ok=True)
    plot_mean_fields_and_abs_grads(sorted_results, save_path=f"{output_dir}/mean_inputs_and_gradients.png")
    return sorted_results

def plot_mean_fields_and_abs_grads(results, save_path=None):
    # Gather data
    inputs_ML_all, inputs_KK_all, grads_ML_all, grads_KK_all = [], [], [], []

    for r in results:
        inputs_ML_all.append(r["mean_inputs_ML"][0])
        inputs_KK_all.append(r["mean_inputs_KK"][0])
        grads_ML_all.append(r["mean_gradient_ML"][0])
        grads_KK_all.append(r["mean_gradient_KK"][0])

    # Convert to arrays
    inputs_ML_all = np.stack(inputs_ML_all)
    inputs_KK_all = np.stack(inputs_KK_all)
    grads_ML_all = np.stack(grads_ML_all)
    grads_KK_all = np.stack(grads_KK_all)

    # Compute mean and mean(abs)
    inputs_ML_mean = np.mean(inputs_ML_all, axis=0)
    inputs_KK_mean = np.mean(inputs_KK_all, axis=0)
    grads_ML_mean = np.mean(grads_ML_all, axis=0)
    grads_KK_mean = np.mean(grads_KK_all, axis=0)
    grads_ML_absmean = np.mean(np.abs(grads_ML_all), axis=0)
    grads_KK_absmean = np.mean(np.abs(grads_KK_all), axis=0)

    # Prepare plot fields
    data_fields = [
        inputs_ML_mean, inputs_KK_mean,
        grads_ML_mean, grads_KK_mean,
        grads_ML_absmean, grads_KK_absmean
    ]
    titles = [
        "Mean Input: TAU-ML", "Mean Input: KK2000",
        "Mean Grad: TAU-ML", "Mean Grad: KK2000",
        "Mean |Grad|: TAU-ML", "Mean |Grad|: KK2000"
    ]
    cmaps = ['viridis', 'viridis', 'coolwarm', 'coolwarm', 'Reds', 'Reds']
    norms = [
        Normalize(vmin=-0.3, vmax=0.3), Normalize(vmin=-0.3, vmax=0.3),
        Normalize(vmin=-1e-6, vmax=1e-6), Normalize(vmin=-1e-6, vmax=1e-6),
        Normalize(vmin=0.0, vmax=1e-5), Normalize(vmin=0.0, vmax=1e-5)
    ]

    # Set up 3x2 grid
    fig = plt.figure(figsize=(14, 12), dpi=150)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.25, wspace=0.15)

    axes, images = [], []
    for i in range(6):
        ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon_grid, lat_grid, data_fields[i],
                           cmap=cmaps[i], norm=norms[i],
                           transform=ccrs.PlateCarree(), shading='auto')
        ax.set_title(titles[i], fontsize=10)
        ax.coastlines()
        ax.gridlines(draw_labels=True, color='gray', linewidth=0.3)
        axes.append(ax)
        images.append(im)

    # Add colorbars (one for inputs, one for grads)
    cbar_ax1 = fig.add_axes([0.91, 0.67, 0.015, 0.25])  # top
    cbar_ax2 = fig.add_axes([0.91, 0.36, 0.015, 0.25])  # middle
    cbar_ax3 = fig.add_axes([0.91, 0.05, 0.015, 0.25])  # bottom

    cb1 = plt.colorbar(images[0], cax=cbar_ax1)
    cb1.set_label("Input Value", fontsize=10)
    cb1.ax.tick_params(labelsize=8)

    cb2 = plt.colorbar(images[2], cax=cbar_ax2)
    cb2.set_label("Gradient Relevance", fontsize=10)
    cb2.ax.tick_params(labelsize=8)

    cb3 = plt.colorbar(images[4], cax=cbar_ax3)
    cb3.set_label("Mean |Gradient|", fontsize=10)
    cb3.ax.tick_params(labelsize=8)

    # Title and layout
    fig.suptitle("Mean and Mean Gradient Magnitude Across PPE Members and Weeks", fontsize=16, y=0.98)
    plt.subplots_adjust(left=0.05, right=0.88, top=0.94, bottom=0.03)
    
    # 🔸 Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()

