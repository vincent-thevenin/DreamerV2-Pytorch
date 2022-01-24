import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(
        self,
        gamma,
        num_action=9,
        has_noise = False,
        noise_std = 0.1,
        is_sample_average = False
    ):
        super(WorldModel, self).__init__()
        self.gamma = gamma  # discount factor
        self.num_action = num_action

        # Experiments
        self.has_noise = has_noise
        self.noise_std = noise_std

        self.is_sample_average = is_sample_average


        #Recurrent Model (RSSM): ((z, a), h) -> h
        self.gru_mlp = nn.Sequential(
            nn.Linear(
                in_features=32 * 32 + num_action,
                out_features=600,
            ),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRUCell(
            input_size=600,
            hidden_size=600,
        )

        #q (1st part): (x) -> xembedded
        self.representation_model_conv = nn.Sequential(
            #1,64,64
            #depth = 2 ** i * default_cnn_depth
            nn.Conv2d(1, 48, 4, padding=0, stride=2), #48,31,31
            nn.ELU(inplace=True),
            nn.Conv2d(48, 2 * 48, 4, padding=0, stride=2), #2 * 48,14,14
            nn.ELU(inplace=True),
            nn.Conv2d(2 * 48, 4 * 48, 4, padding=0, stride=2), #4 * 48, 6, 6
            nn.ELU(inplace=True),
            nn.Conv2d(4 * 48, 8 * 48, 4, padding=0, stride=2), #8 * 48, 2, 2
            nn.ELU(inplace=True),
        )
        #q (2nd part): (h, xembedded) -> z
        self.representation_model_mlp = nn.Sequential(
            nn.Linear(8 * 48 * 2 * 2 + 600, 600),
            nn.ELU(inplace=True),
            nn.Linear(600, 32 * 32),
        )

        #p: (h) -> z_hat
        self.transition_predictor = nn.Sequential(
            nn.Linear(600, 600),
            nn.ELU(inplace=True),
            nn.Linear(600, 32 * 32),
        )

        #p: (h, z) -> r_hat
        self.r_predictor_mlp = nn.Sequential(
            nn.Linear(600 + 32 * 32, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
            nn.Tanh(),
        )

        #p: (h, z) -> gamma_hat
        self.gamma_predictor_mlp = nn.Sequential(
            nn.Linear(600 + 32 * 32, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
        )

        #p: (h,z) -> x_hat
        self.x_hat_predictor_mlp = nn.Sequential(
            nn.Linear(600 + 32 * 32, 32 * 48),
        )
        self.image_predictor_conv = nn.Sequential( #32*48, 1, 1
            nn.ConvTranspose2d(32 * 48, 2**(4 - 0 - 2) * 48, 5, stride=2, padding=0, output_padding=0), #192, 8, 8 (channel = 2**(4-i-2) * 48)
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(2**(4 - 0 - 2) * 48, 2**(4 - 1 - 2) * 48, 5, stride=2, padding=0, output_padding=0), #16, 16, 16
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(2**(4 - 1 - 2) * 48, 2**(4 - 2 - 2) * 48, 6, stride=2, padding=0, output_padding=0), #8, 32, 32
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(2**(4 - 2 - 2) * 48, 1, 6, stride=2, padding=0, output_padding=0), #1, 64, 64
        )
    
    def compute_h(self, batch_size, device, a=None, h=None, z=None):
        """
        In:
            batch_size: int
            device: torch.device of the model
            a: a_t-1
            h: h_t-1
            z: z_t-1
        Out:
            h: h_t
        """
        if h is None: #starting new sequence
            h = torch.zeros((batch_size, 600)).to(device)
        else:
            h = self.gru(
                self.gru_mlp(
                    torch.cat((z.reshape(-1, 32*32), a), dim=1)
                ),
                h
            )

        return h

    def compute_z(self, x, h, k=1, has_grad=True):
        """
        In:
            x: x_t
            h: h_t
        Out:
            z_logit:  logits of z_t
            z_sample: z_t
        """
        embedding = self.representation_model_conv(x)
        embedding = embedding.reshape(x.shape[0], -1)

        embedding = torch.cat((h, embedding), dim=1)
        z_logits = self.representation_model_mlp(embedding)
        z_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_logits.reshape(-1, 32, 32)
        ).sample([k])

        z_sample = z_sample.sum(dim=0)  # sample([k]) returned a list of k samples
        if not self.is_sample_average:
            z_sample = torch.clamp_max(z_sample, 1)
        else:
            z_sample = z_sample / k

        if self.has_noise:
            z_sample += torch.randn_like(z_sample) * self.noise_std

        if has_grad:
            z_probs = torch.softmax(z_logits.reshape(-1, 32, 32), dim=-1)
            z_sample = z_sample + z_probs - z_probs.detach()

        return z_logits, z_sample
    
    def compute_z_hat_sample(self, z_hat_logits, k=1):
        """
        In:
            z_hat_logits: logits of z_hat_t
        Out:
            z_hat_sample (with straight through gradient)
        """
        z_hat_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_hat_logits.reshape(-1, 32, 32)
        ).sample([k])
        
        z_hat_sample = z_hat_sample.sum(dim=0)  # sample([k]) returned a list of k samples
        if not self.is_sample_average:
            z_hat_sample = torch.clamp_max(z_hat_sample, 1)
        else:
            z_hat_sample = z_hat_sample / k

        if self.has_noise:
            z_hat_sample += torch.randn_like(z_hat_sample) * self.noise_std
        
        z_hat_probs = torch.softmax(z_hat_logits.reshape(-1, 32, 32), dim=-1)
        
        return z_hat_sample + z_hat_probs - z_hat_probs.detach()

    def compute_x_hat(self, h_z):
        """
        In:
            h_z: concat of h_t and z_t
        Out:
            x_hat: x_hat_t
        """
        x_hat = self.x_hat_predictor_mlp(h_z)
        x_hat = x_hat.reshape(-1, 32 * 48, 1, 1)
        return self.image_predictor_conv(x_hat)

    #using inference
    def forward_inference(self, a, x, z, h, k=1):
        """
        In:
            a: a_t-1
            x: x_t
            z: z_t-1 (sampled not logits)
            h: h_t-1
        """
        h = self.compute_h(x.shape[0], x.device, a, h, z)

        #no straight-though gradient
        z_logits, z_sample = self.compute_z(x, h, k, has_grad=False)

        return z_sample, h

    def dream(self, a, x, z, h, k=1):
        h = self.compute_h(a.shape[0], a.device, a, h, z)

        z_logits, z_sample = None, None #No z

        z_hat_logits = self.transition_predictor(h)
        z_hat_sample = self.compute_z_hat_sample(z_hat_logits, k)

        x_hat = None

        h_z = torch.cat((h, z_hat_sample.reshape(-1, 32*32)), dim=1)
        r_hat = self.r_predictor_mlp(h_z)
        gamma_hat = self.gamma_predictor_mlp(h_z)

        # r_hat_sample = torch.distributions.normal.Normal(
        #     loc=r_hat,
        #     scale=1.0
        # ).sample()
        r_hat_sample = r_hat.detach()

        gamma_hat_sample = torch.distributions.bernoulli.Bernoulli(
            logits=gamma_hat
        ).sample() * self.gamma #Bernoulli in {0,1}

        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)

    def train(self, a, x, z, h, k=1):
        h = self.compute_h(x.shape[0], x.device, a, h, z)

        z_logits, z_sample = self.compute_z(x, h, k, has_grad=True)

        z_hat_logits = self.transition_predictor(h)
        z_hat_sample = None

        h_z = torch.cat((h, z_sample.reshape(-1, 32*32)), dim=1)

        r_hat = self.r_predictor_mlp(h_z)

        gamma_hat = self.gamma_predictor_mlp(h_z)

        x_hat = self.compute_x_hat(h_z)

        r_hat_sample, gamma_hat_sample = None, None
    
        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)


    def forward(self, a, x, z, h=None, dream=False, inference=False, k=1):
        """
        Input:
            a: a_t-1
            x: x_t
            z: z_t-1
            h: h_t-1
        """
        if inference: #only use embedding network, i.e. no image predictor
            return self.forward_inference(a, x, z, h, k)
        elif dream:
            return self.dream(a, x, z, h, k)
        else:
            return self.train(a, x, z, h, k)

class Actor(nn.Module):
    def __init__(self, num_actions=9):
        super(Actor, self).__init__()
        
        self.num_actions = num_actions

        self.model = nn.Sequential(
            nn.Linear(32*32, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, self.num_actions),
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32*32, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)

class LossModel(nn.Module):
    def __init__(self, nx=1/64/64, nr=1, ng=1, nt=0.08, nq=0.02):
        super(LossModel, self).__init__()

        self.nx = nx
        self.nr = nr
        self.ng = ng
        self.nt = nt
        self.nq = nq

        self.kl = nn.KLDivLoss(log_target=True, reduction='mean')
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, r, gamma, x_hat, r_hat, gamma_hat, z_logits, z_hat_logits):
        x_dist = torch.distributions.normal.Normal(
            loc=x_hat,
            scale=1.0
        )
        r_dist = torch.distributions.normal.Normal(
            loc=r_hat,
            scale=1.0
        )
        gamma_dist = torch.distributions.bernoulli.Bernoulli(
            logits=gamma_hat
        )
        z_logprob = self.log_softmax(z_logits.reshape(-1, 32, 32))
        z_hat_logprob = self.log_softmax(z_hat_logits.reshape(-1, 32, 32))

        loss_x = - self.nx * x_dist.log_prob(x).mean(0).sum()
        loss_r = - self.nr * r_dist.log_prob(r).mean(0).sum()
        loss_g = - self.ng * gamma_dist.log_prob(gamma.round()).mean()
        loss_kl = self.nt * self.kl(z_logprob.detach(), z_hat_logprob)\
            + self.nq * self.kl(z_logprob, z_hat_logprob.detach())
        loss = loss_x + loss_r + loss_g + loss_kl
        return loss, {
            'loss_x':loss_x.item(),
            'loss_r':loss_r.item(),
            'loss_g':loss_g.item(),
            'loss_kl':loss_kl.item(),
        }

class ActorLoss(nn.Module):
    def __init__(self, ns=0.9, nd=0.1, ne=1e-2):
        super(ActorLoss, self).__init__()

        self.ns = 1#ns
        self.nd = 0#nd
        self.ne = ne

        self.anneal = 0#1e-5

    def forward(self, a, dist_a, V, ve):
        loss_rf = (-self.ns * dist_a.log_prob(a) * (V - ve).detach().squeeze(-1)).mean()
        loss_st = (-self.nd * V.squeeze(-1)).mean()
        loss_e = (-self.ne * dist_a.entropy()).mean()

        # self.nd = max(0, self.nd - self.anneal)
        # self.ne = max(3e-4, self.ne - self.anneal)

        loss = loss_rf + loss_st + loss_e

        return loss, {
            'loss_rf': loss_rf.item(),
            'loss_st': loss_st.item(),
            'loss_e': loss_e.item(),
        }

class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, V, ve):
        loss_mse = self.mse(V, ve)
        return loss_mse, {
            'loss_mse': loss_mse.item(),
        }
