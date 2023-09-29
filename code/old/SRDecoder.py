class SRDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        # Create the generator
        self.gen_cell = ClippedGRUCell(
            hps.ext_input_dim + hps.co_dim + hps.com_dim, hps.gen_dim, clip_value=hps.cell_clip
        )
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        init_linear_(self.fac_linear)
        # Create the dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                2 * hps.ci_enc_dim + hps.fac_dim, hps.con_dim, clip_value=hps.cell_clip
            ).float()
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            init_linear_(self.co_linear)
        # Keep track of the state dimensions
        self.state_dims = [
            hps.gen_dim,
            hps.con_dim,
            hps.co_dim,
            hps.co_dim,
            hps.co_dim + hps.ext_input_dim,
            hps.fac_dim,
            hps.com_dim, 
        ]
        # Keep track of the input dimensions
        self.input_dims = [2 * hps.ci_enc_dim, hps.ext_input_dim]

    def forward(self, input, h_0, sample_posteriors=True):
        hps = self.hparams
        
        con_state, gen_state, factor = torch.split(h_0, [hps.con_dim, hps.gen_dim, hps.fac_dim], dim=2)
        ci_step, com_step = torch.split(input, [hps.ci_enc_dim, hps.com_dim])
        
        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor, com_step = torch.split(
            h_0, self.state_dims, dim=1
        )
        ci_step, ext_input_step = torch.split(input, self.input_dims, dim=1)

        if self.use_con:
            # Compute controller inputs with dropout
            con_input = torch.cat([ci_step, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            # Compute and store the next hidden state of the controller
            con_state = self.con_cell(con_input_drop.float(), con_state.float()) # TODO
            # Compute the distribution of the controller outputs at this timestep
            co_params = self.co_linear(con_state)
            co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar))
            # Sample from the distribution of controller outputs
            co_post = self.hparams.co_prior.make_posterior(co_mean, co_std)
            con_output = co_post.rsample() if sample_posteriors else co_mean
            # Combine controller output with any external inputs
            gen_input = torch.cat([con_output, ext_input_step, com_step], dim=1)
        else:
            # If no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state = self.gen_cell(gen_input.float(), gen_state.float())
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_std, gen_input, factor], dim=1
        )

        return hidden