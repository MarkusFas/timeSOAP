class SOAP_descriptor():
    def __init__(self, HYPERS, selected_atoms, centers, neighbors):
        self.calculator = SoapPowerSpectrum(**HYPERS)

        self.sel_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )

        self.sel_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def calculate(self, systems):
        
        soap = self.calculator(
            systems,
            selected_samples=self.sel_samples,
            selected_keys=self.sel_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        soap_block = soap.block()
        return soap_block