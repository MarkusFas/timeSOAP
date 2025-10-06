

class SOAP_descriptor():
    def __init__(self, HYPERS, selected_atoms, centers, neighbors):
        self.calculator = SoapPowerSpectrum(**HYPERS)

        self.sel = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )

        self.atomsel = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def calculate(self):

class FullMethod():

    def __init__(self, descriptor, selected_atoms, interval, lag):
        self.interval = interval
        self.lag = lag
        self.selected_atoms = selected_atoms
        self.descriptor = descriptor
        

    def compute_COV(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block.values.numpy()  
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in centers]
    
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]).values.numpy()
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
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

            buffer[:,fidx%self.interval,:] = new_soap_values

        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        return mean_mu_t, mean_cov_t, cov_mu_t, atomsel_element
