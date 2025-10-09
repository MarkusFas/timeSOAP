import torch 
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import ase.neighborlist
from vesin import ase_neighbor_list
from memory_profiler import profile

def eval_PETMAD(structures, atomsel):
    """
    get  last layer PET features for a PET-MAD model as descriptors
    add slicing logic to select for subset of atoms to be inline with 
    SOAP implementation
    """
    petmad = load_atomistic_model('/Users/markusfasching/EPFL/Work/project-CVs/La-dota/aims-prep/combined-data/new/pet-mad-latest.pt')
    systems = systems_to_torch(structures)
    nl_options = petmad.requested_neighbor_lists()[0]

    for i, system in enumerate(systems):
        atoms = structures[i]
        i, j, S, D = ase_neighbor_list(quantities="ijSD", a=atoms, cutoff=4.5)
        i = torch.from_numpy(i.astype(int))
        j = torch.from_numpy(j.astype(int))
        neighbor_indices = torch.stack([i, j], dim=1)
        neighbor_shifts = torch.from_numpy(S.astype(int))
        sample_values = torch.hstack([neighbor_indices, neighbor_shifts])
        samples = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=sample_values,
        )

        neighbors = TensorBlock(
            values=torch.from_numpy(D).reshape(-1, 3, 1),
            samples=samples,
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        ).to(torch.float32)
        system.add_neighbor_list(nl_options, neighbors)

    output = ModelOutput(
        quantity='', # mtt::aux::energy_last_layer_features
        unit='',
        per_atom=True,
        explicit_gradients=[],
    )
    output_energy = ModelOutput(
        quantity='energy', # mtt::aux::energy_last_layer_features
        unit='eV',
        #per_atom=True,
        explicit_gradients=[],
    )
    options = ModelEvaluationOptions(
        length_unit='angstrom', 
        outputs={
            'mtt::aux::energy_last_layer_features': output, 
            #'energy': output_energy,
        }, # check features, check 'mtt::aux::energy_last_layer_features'
        #selected_atoms=atomsel, # doesnt work anyway, if we want to use it,
        #  we need to either implement it with view in metatrain or
        # simply add a system dimenion in the labels...
    )

    pet_mad_features = petmad(systems, 
                options=options,
                check_consistency=True,
    )
    features = pet_mad_features['mtt::aux::energy_last_layer_features']
    #energies = pet_mad_features['energy']
    #energies = energies[0].values.detach().numpy()
    #energies_per_atom = energies/np.array(n_atoms)
    #feat = pet_mad_features['mtt::aux::energy_last_layer_features'].samples_to_keys('atom')
    new_map = mts.split(
            features,
            axis="samples",
            selections=[
                atomsel,
            ],
        )
    #feat = mean_over_samples(new_map[0], sample_names=["atom"]) 
    
    return new_map[0].block().values.numpy() # T*N, P #.reshape(len(systems), len(atomsel.values), -1) #T, N, P

def get_SOAP(traj, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    systems = systems_to_torch(traj, dtype=torch.float64)
    all_soaps = []
    soap = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    return soap.reshape(len(traj),-1, soap.shape[-1]) #T, N, P


def eval_SOAP(systems, calculator, sel, atomsel):
    soap = calculator(
            systems,
            selected_samples=atomsel,
            selected_keys=sel,
        )
    soap = soap.keys_to_samples("center_type")
    #soap = soap.keys_to_properties(soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"]))
    soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    soap_block = soap.block()
    #soap_values = soap_block.values.numpy()
    return soap_block


def SOAP_PCA(traj, ids_atoms, HYPER_PARAMETERS, centers, neighbors, pcatrafo):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    systems = systems_to_torch(traj, dtype=torch.float64)
    pca = []
    for i, system in enumerate(systems):
        soap = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        #petfeatures = eval_PETMAD([traj[i]], atomsel)
        pca.append(pcatrafo.trafo(soap))
    
    print(len(pca))
    pca = np.stack(pca, axis=0)
    
    return pca.transpose(1,0,2) # should be N, T, P

@profile
def SOAP_full(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    inframe = np.zeros((len(atomsel_element), len(systems), first_soap.shape[0], first_soap.shape[1]))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element), dtype=int)
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        if fidx >= maxlag:
            roll_kernel = np.roll(kernel, fidx%maxlag)
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                inframe[atom_type_idx, ntimesteps[atom_type_idx]] = avg_soap[atom_type]
                ntimesteps[atom_type_idx] += 1

        buffer[:,fidx%maxlag,:] = new_soap_values
    return [soaps[:ntimesteps[atom_type_idx]] for atom_type_idx, soaps in enumerate(inframe)], soap_block.properties # trim zeros from the back 


@profile
def SOAP_mean(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    inframe = np.zeros((len(atomsel_element), len(systems), first_soap.shape[1]))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element), dtype=int)
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        if fidx >= maxlag:
            roll_kernel = np.roll(kernel, fidx%maxlag)
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                inframe[atom_type_idx, ntimesteps[atom_type_idx]] = avg_soap[atom_type].mean(axis=0)
                ntimesteps[atom_type_idx] += 1

        buffer[:,fidx%maxlag,:] = new_soap_values
    return [soaps[:np.where(soaps==0)[0][0],:] for soaps in inframe], soap_block.samples.values.numpy() # trim zeros from the back 



@profile
def SOAP_COV_atomsprops(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    #test_ = eval_PETMAD(traj, atomsel)

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    avgcov = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    soapsum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    inframe_means = np.zeros((len(atomsel_element),first_soap.shape[1], first_soap.shape[1],))
    inframe_sum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element))
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):

                inframe = avg_soap[atom_type].mean(axis=0)
                inframe_sum[atom_type_idx] += inframe
                #soapsum_lag[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                # ge and te averaged covariance
                # PCA avgcov[atom_type_idx] += np.einsum("ia,ib->ab", new_soap_values[atom_type], new_soap_values[atom_type]) #sum over all same atoms (have already summed over all times before)
                avgcov[atom_type_idx] += np.einsum("a,b->ab", inframe, inframe) 
                nsmp[atom_type_idx] += len(atom_type)
                ntimesteps[atom_type_idx] += 1

        buffer[:,fidx%maxlag,:] = new_soap_values

    avgcc = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    avgtcov = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    avgmean = np.zeros((len(atomsel_element), first_soap.shape[1],))
    avgmean_lag = np.zeros((len(atomsel_element), first_soap.shape[1],))


    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        avgmean[atom_type_idx] = inframe_sum[atom_type_idx]/ntimesteps[atom_type_idx] #soapsum[atom_type_idx, :, None]/nsmp[atom_type_idx]
        #avgmean_lag[atom_type_idx] = soapsum[atom_type_idx, :]/nsmp[atom_type_idx]
        avgcc[atom_type_idx] = avgcov[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', avgmean[atom_type_idx], avgmean[atom_type_idx]) #soapsum[atom_type_idx, :, None]*soapsum[atom_type_idx, None, :]/(nsmp[atom_type_idx]**2)
        # COV = 1/N ExxT - mumuT

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))
    return avgmean, avgcc, atomsel_element



@profile
def SOAP_COV_repair(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    #test_ = eval_PETMAD(traj, atomsel)

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    avgcov = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    soapsum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    inframe_means = np.zeros((len(atomsel_element),first_soap.shape[1], first_soap.shape[1],))
    inframe_sum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element))
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):

                inframe = avg_soap[atom_type].mean(axis=0)
                inframe_sum[atom_type_idx] += inframe
                #soapsum_lag[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                # ge and te averaged covariance
                # PCA avgcov[atom_type_idx] += np.einsum("ia,ib->ab", new_soap_values[atom_type], new_soap_values[atom_type]) #sum over all same atoms (have already summed over all times before)
                avgcov[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) 
                nsmp[atom_type_idx] += len(atom_type)
                ntimesteps[atom_type_idx] += 1

        buffer[:,fidx%maxlag,:] = new_soap_values

    avgcc = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    avgtcov = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    avgmean = np.zeros((len(atomsel_element), first_soap.shape[1],))
    avgmean_lag = np.zeros((len(atomsel_element), first_soap.shape[1],))


    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        avgmean[atom_type_idx] = inframe_sum[atom_type_idx]/ntimesteps[atom_type_idx] #soapsum[atom_type_idx, :, None]/nsmp[atom_type_idx]
        #avgmean_lag[atom_type_idx] = soapsum[atom_type_idx, :]/nsmp[atom_type_idx]
        avgcc[atom_type_idx] = avgcov[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', avgmean[atom_type_idx], avgmean[atom_type_idx]) #soapsum[atom_type_idx, :, None]*soapsum[atom_type_idx, None, :]/(nsmp[atom_type_idx]**2)
        # COV = 1/N ExxT - mumuT

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))
    return avgmean, avgcc, atomsel_element


@profile
def SOAP_COV_directPCAtestfull(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )


    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    cov_t = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    sum_mu_t = np.zeros((len(atomsel_element),first_soap.shape[1],))
    scatter_mut = np.zeros((len(atomsel_element),first_soap.shape[1], first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element), dtype=int)
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    inframe = np.zeros((len(atomsel_element), len(systems), first_soap.shape[1]))
    X = []
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                mu_t = avg_soap[atom_type].mean(axis=0)
                scatter_mut[atom_type_idx] += np.einsum(
                    "a,b->ab", 
                    mu_t, 
                    mu_t,
                )  
                X.append(avg_soap[atom_type].reshape(1, -1, first_soap.shape[1]))
                sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms
                inframe[atom_type_idx, ntimesteps[atom_type_idx]] = mu_t
                cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                nsmp[atom_type_idx] += len(atom_type)
                ntimesteps[atom_type_idx] += 1


        buffer[:,fidx%maxlag,:] = new_soap_values

    X = np.concatenate(X, axis=1)
    mean_cov_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    cov_mu_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    mean_mu_t = np.zeros((len(atomsel_element), first_soap.shape[1],))

    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        
        mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
        # COV = 1/N ExxT - mumuT
        mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
        # add temporal covariance
        cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))


    X_train = [soaps[:np.where(soaps==0)[0][0],:] for soaps in inframe]
    X_train = [x for x in X]
    return mean_mu_t, mean_cov_t, cov_mu_t, X_train, atomsel_element
    #return avgmean, mean_cov_t, cov_mu_t, atomsel_element


@profile
def SOAP_COV_directPCAtest(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )


    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    cov_t = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    soap_sum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    scatter_mut = np.zeros((len(atomsel_element),first_soap.shape[1], first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element), dtype=int)
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    inframe = np.zeros((len(atomsel_element), len(systems), first_soap.shape[1]))
    X = []
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                soap_sum += avg_soap[atom_type].sum(axis=0)
                
                X.append(avg_soap[atom_type].reshape(1, -1, first_soap.shape[1]))
                cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                nsmp[atom_type_idx] += len(atom_type)
                ntimesteps[atom_type_idx] += 1


        buffer[:,fidx%maxlag,:] = new_soap_values

    X = np.concatenate(X, axis=1)
    mean_cov_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    cov_mu_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    mean_mu_t = np.zeros((len(atomsel_element), first_soap.shape[1],))

    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        
        mean_mu_t[atom_type_idx] = soap_sum[atom_type_idx]/nsmp[atom_type_idx]
        # COV = 1/N ExxT - mumuT
        
        cov_mu_t[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))


    X_train = [soaps[:np.where(soaps==0)[0][0],:] for soaps in inframe]
    X_train = [x for x in X]
    return mean_mu_t, cov_mu_t, X_train, atomsel_element
    #return avgmean, mean_cov_t, cov_mu_t, atomsel_element



@profile
def SOAP_COV_test(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    cov_t = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    sum_mu_t = np.zeros((len(atomsel_element),first_soap.shape[1],))
    scatter_mut = np.zeros((len(atomsel_element),first_soap.shape[1], first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    ntimesteps = np.zeros(len(atomsel_element), dtype=int)
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                mu_t = avg_soap[atom_type].mean(axis=0)
                scatter_mut[atom_type_idx] += np.einsum(
                    "a,b->ab", 
                    mu_t, 
                    mu_t,
                )  

                sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

                cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                nsmp[atom_type_idx] += len(atom_type)
                ntimesteps[atom_type_idx] += 1

        buffer[:,fidx%maxlag,:] = new_soap_values


    mean_cov_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    cov_mu_t = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    mean_mu_t = np.zeros((len(atomsel_element), first_soap.shape[1],))

    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        
        mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
        # COV = 1/N ExxT - mumuT
        mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
        # add temporal covariance
        cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))

    return mean_mu_t, mean_cov_t, cov_mu_t, atomsel_element
    #return avgmean, mean_cov_t, cov_mu_t, atomsel_element



@profile
def SOAP_TICA_repair(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    #test_ = eval_PETMAD(traj, atomsel)

    systems = systems_to_torch(traj, dtype=torch.float64)
    soap_block = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = soap_block.values.numpy()
    #first_soap = eval_PETMAD(traj[:1], atomsel)
    atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
    maxlag = interval
    buffer = np.zeros((first_soap.shape[0], maxlag, first_soap.shape[1]))
    avgcov = np.zeros((len(atomsel_element), first_soap.shape[1], first_soap.shape[1],))
    soapsum = np.zeros((len(atomsel_element),first_soap.shape[1],))
    #soapsum_lag = np.zeros((len(atomsel_element),first_soap.shape[1],))
    nsmp = np.zeros(len(atomsel_element))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=(maxlag-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
    kernel = np.zeros(maxlag)
    kernel[maxlag//2]=1
    #atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center
    # for the PET case
    #atomsel_element=[[atom.index for atom in traj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    
    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        #new_soap_values = eval_PETMAD([traj[fidx]], atomsel)
        #idx = (fidx + maxlag//2)
        new_soap_values = eval_SOAP([system], calculator, sel, atomsel).values.numpy()
        if fidx >= maxlag:
            #first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            avg_soap_lag = np.einsum("j,ija->ia", np.roll(roll_kernel, maxlag//2), buffer) # to get the lagged values, roll through half the buffer
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                soapsum[atom_type_idx] += avg_soap[atom_type].sum(axis=0) #sum over all same atoms
                #soapsum_lag[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                # ge and te averaged covariance
                # PCA avgcov[atom_type_idx] += np.einsum("ia,ib->ab", new_soap_values[atom_type], new_soap_values[atom_type]) #sum over all same atoms (have already summed over all times before)
                avgcov[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap_lag[atom_type]) 
                nsmp[atom_type_idx] += len(atom_type)

        buffer[:,fidx%maxlag,:] = new_soap_values

    avgcc = np.zeros((len(atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
    avgmean = np.zeros((len(atomsel_element), first_soap.shape[1],))
    avgmean_lag = np.zeros((len(atomsel_element), first_soap.shape[1],))
    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        avgmean[atom_type_idx] = soapsum[atom_type_idx, :]/nsmp[atom_type_idx] #soapsum[atom_type_idx, :, None]/nsmp[atom_type_idx]
        #avgmean_lag[atom_type_idx] = soapsum[atom_type_idx, :]/nsmp[atom_type_idx]
        avgcc[atom_type_idx] = avgcov[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', avgmean[atom_type_idx], avgmean[atom_type_idx]) #soapsum[atom_type_idx, :, None]*soapsum[atom_type_idx, None, :]/(nsmp[atom_type_idx]**2)
        # COV = 1/N ExxT - mumuT
    

    #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
    #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
    #print(np.allclose(C_np, avgcc[0], atol=1e-8))
    return avgmean, avgcc, avgcovariance, atomsel_element


@profile
def SOAP_COV(traj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors):
    # select which atoms to compute the SOAP for (here all)
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

    sel = Labels(
        names=["center_type", "neighbor_1_type", "neighbor_2_type"],
        values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
            k], dtype=torch.int32),
    )

    atomsel = Labels(
        names=["atom"],
        values=torch.tensor(ids_atoms, dtype=torch.int64).unsqueeze(-1),
    )

    #test_ = eval_PETMAD(traj, atomsel)

    systems = systems_to_torch(traj, dtype=torch.float64)
    #first_soap = eval_SOAP(systems[:1], calculator, sel, atomsel)
    first_soap = eval_PETMAD(traj[:1], atomsel)
    
    maxlag = interval
    buffer = np.zeros((atomsel.values.shape[0], maxlag, first_soap.shape[1]))
    avgcov = np.zeros((atomsel.values.shape[0], first_soap.shape[1], first_soap.shape[1],))
    soapsum = np.zeros((atomsel.values.shape[0],first_soap.shape[1],))
    
    nsmp = np.zeros(len(atomsel))
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=maxlag//7)


    atomsel_element=[np.arange(atomsel.values.shape[0]) for _ in centers] #one entry for each SOAP center


    for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
        new_soap_values = eval_PETMAD([traj[fidx]], atomsel)

        if fidx>=maxlag:
            first = buffer[:,fidx%maxlag] # takes first/ oldest soap in buffer (maxlag timesteps ago)
            roll_kernel = np.roll(kernel, -fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
            # could loop over it like hannah, but dont know why yet...I think it chooses an atom type and averages over all atoms of same type to get PCA/COV
            atomsel_idx = np.arange(atomsel.values.shape[0]) 
            for atom_type_idx, atom_type in enumerate(atomsel_element):
                soapsum[atom_type_idx] += avg_soap[atom_type].sum(axis=0) #sum over all same atoms
                # ge and te averaged covariance
                # PCA avgcov[atom_type_idx] += np.einsum("ia,ib->ab", new_soap_values[atom_type], new_soap_values[atom_type]) #sum over all same atoms (have already summed over all times before)
                avgcov[atom_type_idx] += np.einsum("ia,ib->ab", new_soap_values[atom_type], first[atom_type]) 
                nsmp[atom_type_idx] += len(atom_type)

        buffer[:,fidx%maxlag,:] = new_soap_values

    avgcc=np.zeros((len(atomsel_element), new_soap_values.shape[1],new_soap_values.shape[1]))
    # autocorrelation matrix - remove mean
    for atom_type_idx, atom_type in enumerate(atomsel_element):
        avgcc[atom_type_idx] = avgcov[atom_type_idx]/nsmp[atom_type_idx] - soapsum[atom_type_idx, :, None]*soapsum[atom_type_idx, None, :]/(nsmp[atom_type_idx]**2)
    return avgcc

def setup_soap_predictor(atomsel,systems):
    print(atomsel)
    print(systems)
    print(torch.tensor(
            [
                [j, i]
                #for i in systems[0].types
                for i in atomsel
                for j in range(len(systems))
                # if j % 100 == 0
            ]
        ).shape)
    selected_atoms = Labels(
        ["system", "atom"],
        torch.tensor(
            [
                [j, i]
                #for i in systems[0].types
                for i in atomsel
                for j in range(len(systems))
                # if j % 100 == 0
            ]
        ),
    )
    #print(systems[0].types)
    #print(torch.unique(systems[0].types))
    species=torch.unique(systems[0].types).tolist()
    print(species)
    soap = SOAP_pred(species=species, hypers=hypers)
    soap.eval()    
    capabilities = ModelCapabilities(
        outputs={"features": ModelOutput(per_atom=True)},
        interaction_range=10.0,
        supported_devices=["cpu"],
        length_unit="A",
        atomic_types=species,
        dtype="float64",
    )
    
    metadata = ModelMetadata(name="SOAP water")
    model_soap = AtomisticModel(soap, metadata, capabilities)
    #model.save("soap_cv.pt", collect_extensions="extensions")
    
    #get soap calculated by model
    opts = ModelEvaluationOptions(
        length_unit="A",
        outputs={"features": ModelOutput(quantity="", per_atom=True)},
        selected_atoms=selected_atoms,
    )
    return model_soap, opts


def compute_autocorrelation_average(systems, hypers, kernel, atomsel, atomsel_element, maxlag=100):
    print('computeautocorrelation_average')
    print('atomsel',atomsel)
    print('atomsel_element',atomsel_element)
    model_soap,opts=setup_soap_predictor(atomsel,systems)
    soap_pred = model_soap.forward(systems, options=opts, check_consistency=False)#, selected_keys=selection_O)
    rho2i_values=soap_pred['features'][0].values
    #calculator = featomic.torch.SoapPowerSpectrum(**hypers)
    #rho2i_values=compute_soap_to_values(systems, calculator)
    
    buffer = np.zeros((len(atomsel), maxlag, rho2i_values.shape[1]))
    avgcov = np.zeros((len(atomsel_element), rho2i_values.shape[1], rho2i_values.shape[1],))
    soapsum = np.zeros((len(atomsel_element) ,rho2i_values.shape[1],))
    
    nsmp = np.zeros(len(atomsel))
    for fidx, system in tqdm(enumerate(systems)):
    
        #rho2i = calculator.compute(system) # , selected_keys=selection)
        #rho2i = rho2i.keys_to_samples(["center_type"]).keys_to_properties(
        #    ["neighbor_1_type", "neighbor_2_type"]
        #)
        #new_soap = rho2i.block(0).values[atomsel]
        soap_pred = model_soap.forward([system], options=opts, check_consistency=False)#, selected_keys=selection_O)
        new_soap=soap_pred['features'][0].values

        if fidx>=maxlag:
            first = buffer[:,fidx%maxlag]
            roll_kernel = np.roll(kernel, -fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer)
            for ielem,elem in enumerate(atomsel_element):
                 soapsum[ielem] += avg_soap[elem].sum(axis=0)
                 # ge and te averaged covariance
                 avgcov[ielem] += np.einsum("ia,ib->ab", new_soap[elem], new_soap[elem])
                 nsmp[ielem] += len(elem)
    
        buffer[:,fidx%maxlag] = new_soap

    avgcc=np.zeros((len(atomsel), new_soap.shape[1],new_soap.shape[1]))
    # autocorrelation matrix - remove mean
    for ielem,elem in enumerate(atomsel_element):
        avgcc[ielem] = avgcov[ielem]/nsmp[ielem] - soapsum[ielem, :, None]* soapsum[ielem, None, :]/(nsmp[ielem]**2)
    return avgcc


if __name__=='__main__':
    print('Nothing to do here')