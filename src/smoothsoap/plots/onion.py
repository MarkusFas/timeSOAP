from matplotlib.gridspec import GridSpec
from tropea_clustering import onion_multi, onion_uni, helpers
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
import ase.io
import mpltex

@mpltex.acs_decorator
def plot_onion(X, atoms, test_atoms, label, delta_t=None, i_pca=0):
    #try:
    n_seq, n_particles, n_dim = X.shape
    for delta_t in list(set(np.logspace(np.log10(2), np.log10(n_seq), 10, dtype=int))):
        if delta_t == 1:
            continue
        data = X.transpose(2,1,0)[i_pca,...]
        # Select time resolution
        reshaped_input_data = helpers.reshape_from_nt(data, delta_t)
        # Run Onion Clustering
        #state_list, labels = onion_multi(reshaped_input_data)
        state_list, labels = onion_uni(reshaped_input_data)
        classes, class_counts = np.unique(labels, return_counts=True)
        text = label + f'_onion_'
        for cls, count in zip(classes, class_counts):
            text += f'_{cls}-{count}'
        np.savetxt(text + '.txt', labels, fmt='%d')
        cmap = plt.get_cmap("Set1")   # or tab20, Set1, etc.

        class_to_color = {
            cls: cmap(i / max(1, len(classes)-1))[:3]
            for i, cls in enumerate(classes)
        }
        colors = np.array([class_to_color[int(c)] for c in classes])
        fig = plt.figure(figsize=(6,3.5))
        gs = GridSpec(1, 2, width_ratios=[4,0.8], wspace=0.05)

        # main timeseries plot
        ax0 = plt.subplot(gs[0])
    
        # vertical histogram on the right
        ax1 = plt.subplot(gs[1], sharey=ax0)
        N, T = data.shape
        data = data.reshape(-1)
        counts, bin_edges = np.histogram(data, bins=100)
        bin_widths = np.diff(bin_edges)[0]
        norm = (counts.sum() * bin_widths)
        pdf = counts / norm
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax1.barh(bin_centers, pdf, height=bin_widths, align="center", alpha=0.5)
        x = np.linspace(np.min(data), np.max(data), 200)
        tot_perc = np.sum([state.perc for state in state_list])
        norm_factor = np.sum([state.area for state in state_list])
        tot_area = np.sum([state.area for state in state_list])
        tot_perc = np.sum([state.perc for state in state_list])
        tot_area *= tot_perc
        proper_classes = classes[classes != -1]
        for i, class_ in enumerate(proper_classes):
            state = state_list[i]
            mu = state.mean
            var = state.sigma**2
            area = (state.area/tot_area)
            y = np.exp(-0.5*((x - mu)**2 / var))/np.sqrt(2*np.pi*var) * area
            scale = state.perc #counts[i]/np.sum(counts)
            color = class_to_color[int(class_)]
            ax1.plot(y, x, c=color, label=fr'{class_}: {scale*100:.1f} $\%$')

            #add histogram
        #ax.hist(data[0], bins=50, alpha=0.2, color='red', density=True)
        #ax1.set_xlabel('PCA 1')
        ax1.set_xlabel('PDF')
        ax1.legend()

        ax0.set_xlabel('time [ns]')
        ax0.set_ylabel(f'CV')
        data = X.transpose(2,1,0)[i_pca,...]
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        #ax0.plot(mean, alpha=1.0, color='C1')
        #ax0.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.25, color='C0')
        for trj in data.reshape(n_particles, -1):
            ax0.plot(np.arange(len(trj))*0.01, trj, alpha=0.05, color='C0')
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.tight_layout()
        plt.savefig(label + f'_onion_states_delta_{delta_t}.png', dpi=200)
        plt.close()
        plot_histogram_onion(x, bin_centers, pdf, bin_widths, proper_classes, state_list, tot_area, class_to_color, label, delta_t)
        print(f"Saved onion states plot with delta_t={delta_t}")
        #plot_snapshot_onion(labels.reshape(n_particles,-1), atoms, test_atoms, label, delta_t, class_to_color)
        plot_snapshot_onion(labels.reshape(n_particles,-1)[:,0], atoms, test_atoms, label, delta_t, class_to_color)


@mpltex.acs_decorator
def plot_histogram_onion(x, bin_centers, pdf, bin_widths, proper_classes, state_list, tot_area, class_to_color, label, delta_t):
    fig, ax = plt.subplots(1,1,figsize=(3,1.75))
    ax.bar(bin_centers, pdf, width=bin_widths, align="center", alpha=0.5)
    for i, class_ in enumerate(proper_classes):
        state = state_list[i]
        mu = state.mean
        var = state.sigma**2
        area = (state.area/tot_area)
        y = np.exp(-0.5*((x - mu)**2 / var))/np.sqrt(2*np.pi*var) * area
        scale = state.perc #counts[i]/np.sum(counts)
        color = class_to_color[int(class_)]
        ax.plot(x, y, c=color, label=fr'{class_}: {scale*100:.1f} $\%$')
    ax.set_ylabel('PDF')
    ax.set_xlabel(f'CV')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(label + f'_onion_histo_only_{delta_t}.png', dpi=200)
    plt.close()

def plot_snapshot_onion(labels, atoms, test_atoms, label, delta_t, class_to_color=None):

    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.data import DataCollection
    from ovito.modifiers import ColorCodingModifier
    from ovito.vis import Viewport, TachyonRenderer, ParticlesVis

    classes = np.unique(labels)
    colors_ = np.array([class_to_color[int(c)] for c in classes])
    # ----------------------------
    # 1. Create example data
    # ----------------------------
    atoms.wrap()
    # TODO generalise for multiple centers
    selected = test_atoms #[i for i, a in enumerate(atoms) if a.symbol == 'C']
    atoms_oxy = atoms[selected].copy()

    # Remove string / unsupported arrays
    for key in list(atoms_oxy.arrays.keys()):
        if atoms_oxy.arrays[key].dtype.kind not in "iuf":
            del atoms_oxy.arrays[key]
    # Convert the ASE object to an OVITO DataCollection:
    N = len(atoms_oxy)
    
    data = ase_to_ovito(atoms_oxy)
    # We may now create a Pipeline object with a StaticSource and use the 
    # converted dataset as input for a data pipeline:
    pipeline = Pipeline(source = StaticSource(data = data))
    # ----------------------------
    # 2. Custom modifier
    # ----------------------------
    def custom_modifier(frame: int, data: DataCollection):

        particles = data.particles_

        # Create or overwrite the Class property
        particles.create_property(
            name='Class',
            data=labels
        )

        colors = np.array([class_to_color[int(c)] for c in labels])
        particles.create_property(
            name='Color',
            data=colors
        )

    # Attach modifier to pipeline
    pipeline.modifiers.append(custom_modifier)


    # ----------------------------
    # 4. Render image
    # ----------------------------
    pipeline.add_to_scene()

    data = pipeline.compute()

    # Particle size
    data.particles.vis.radius = 0.25

    # Show box
    data.cell.vis.enabled = True
    data.cell.vis.line_width = 0.3

    vp = Viewport(type=Viewport.Type.Ortho)

    vp.camera_dir = (0, -1, 0)   # look along -Y
    vp.camera_up  = (0, 0, 1)    # Z is up → XZ plane
    vp.zoom_all()
    vp.render_image(
        filename=label + f'_onion_snapshot_deltat{delta_t}.png',
        size=(2000, 2000),
        renderer=TachyonRenderer()
    )
    pipeline.remove_from_scene()
    del pipeline


def plot_snapshot(X, atoms, test_atoms, label, i_pca=0):

    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.data import DataCollection
    from ovito.modifiers import ColorCodingModifier
    from ovito.vis import Viewport, TachyonRenderer, ParticlesVis

    from matplotlib import colors
    data = X[...,i_pca] # first PC
    norm = colors.Normalize(vmin=data.min(), vmax=data.max())
    normalized_values = norm(data[0]) # plot only first frame 
    cmap = plt.get_cmap("RdYlBu")   # or tab20, Set1, etc.

    colors_ = cmap(normalized_values)[:,:3]
# ----------------------------
    # 1. Create example data
    # ----------------------------
    atoms.wrap()
    #TODO generalise for multiple centers
    selected = test_atoms #[i for i, a in enumerate(atoms) if a.symbol == 'O']
    atoms_oxy = atoms[selected].copy()

    # Remove string / unsupported arrays
    for key in list(atoms_oxy.arrays.keys()):
        if atoms_oxy.arrays[key].dtype.kind not in "iuf":
            del atoms_oxy.arrays[key]
    # Convert the ASE object to an OVITO DataCollection:
    N = len(atoms_oxy)
    
    data = ase_to_ovito(atoms_oxy)
    # We may now create a Pipeline object with a StaticSource and use the 
    # converted dataset as input for a data pipeline:
    pipeline = Pipeline(source = StaticSource(data = data))
    # ----------------------------
    # 2. Custom modifier
    # ----------------------------
    def custom_modifier(frame: int, data: DataCollection):

        particles = data.particles_

        particles.create_property(
            name='Color',
            data=colors_
        )

    # Attach modifier to pipeline
    pipeline.modifiers.append(custom_modifier)

    # ----------------------------
    # 3. Color particles by class
    # ----------------------------



    # ----------------------------
    # 4. Render image
    # ----------------------------
    pipeline.add_to_scene()

    data = pipeline.compute()

    # Particle size
    data.particles.vis.radius = 0.25

    # Show box
    data.cell.vis.enabled = True
    data.cell.vis.line_width = 0.3

    vp = Viewport(type=Viewport.Type.Ortho)

    vp.camera_dir = (0, -1, 0)   # look along -Y
    vp.camera_up  = (0, 0, 1)    # Z is up → XZ plane
    vp.zoom_all()
    vp.render_image(
        filename=label + f'_snapshot.png',
        size=(2000, 2000),
        renderer=TachyonRenderer()
    )
    pipeline.remove_from_scene()
    del pipeline
    print(f"Saved snapshot")
