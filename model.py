import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, gamma, num_action=9):
        super(WorldModel, self).__init__()
        self.gamma = gamma #discount factor
        self.num_action = num_action

        #Recurrent Model (RSSM): ((z, a), h) -> h
        self.gru = nn.GRUCell(
            input_size=1024 + num_action,
            hidden_size=512,
        )

        #q (1st part): (x) -> xembedded
        self.representation_model_conv = nn.Sequential(
            #1,64,64
            #TODO
            # #512, 1, 1
        )
        #q (2nd part): (h, xembedded) -> z
        self.representation_model_mlp = nn.Sequential(
            #TODO
        )

        #p: (h) -> z_hat
        self.transition_predictor = nn.Sequential(
            #TODO
        )

        #p: (h, z) -> r_hat
        self.r_predictor_mlp = nn.Sequential(
            #TODO
        )

        #p: (h, z) -> z_hat
        self.gamma_predictor_mlp = nn.Sequential(
            #TODO
        )

        #p: (h,z) -> x_hat
        self.x_hat_predictor_mlp = nn.Sequential(
            #TODO
        )
        self.image_predictor_conv = nn.Sequential( #64, 4, 4
            #TODO
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
            h = torch.zeros((batch_size, 512)).to(device)
        else:
            #TODO

        return h

    def compute_z(self, x, h):
        """
        In:
            x: x_t
            h: h_t
        Out:
            z_logit:  logits of z_t
            z_sample: z_t
        """
        #TODO

        return z_logits, z_sample
    
    def compute_z_hat_sample(self, z_hat_logits):
        """
        In:
            z_hat_logits: logits of z_hat_t
        Out:
            z_hat_sample (with straight through gradient)
        """
        #TODO
        
        return z_hat_sample + z_hat_probs - z_hat_probs.detach()

    def compute_x_hat(self, h_z):
        """
        In:
            h_z: concat of h_t and z_t
        Out:
            x_hat: x_hat_t
        """
        #TODO

        return self.image_predictor_conv(x_hat)

    #using inference
    def forward_inference(self, a, x, z, h):
        """
        In:
            a: a_t-1
            x: x_t
            z: z_t-1 (sampled not logits)
            h: h_t-1
        """
        #TODO

        #no straight-though gradient
        return z_sample, h

    def dream(self, a, x, z, h):
        #TODO

        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)

    def train(self, a, x, z, h):
        #TODO
    
        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)


    def forward(self, a, x, z, h=None, dream=False, inference=False):
        """
        Input:
            a: a_t-1
            x: x_t
            z: z_t-1
            h: h_t-1
        """
        if inference: #only use embedding network, i.e. no image predictor
            return self.forward_inference(a, x, z, h)
        elif dream:
            return self.dream(a, x, z, h)
        else:
            return self.train(a, x, z, h)

class Actor(nn.Module):
    def __init__(self, num_actions=9):
        super(Actor, self).__init__()
        
        self.num_actions = num_actions

        self.model = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, num_actions),
            nn.Tanh()
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)*2

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)

class LossModel(nn.Module):
    def __init__(self, nx=1/64/64/3, nr=1, ng=1, nt=0.08, nq=0.1):
        super(LossModel, self).__init__()

        self.nx = nx
        self.nr = nr
        self.ng = ng
        self.nt = nt
        self.nq = nq

    def forward(self, x, r, gamma, z_logits, z_sample, x_hat, r_hat, gamma_hat, z_hat_logits):
        #TODO

        return loss

class ActorLoss(nn.Module):
    def __init__(self, ns=0.9, nd=0.1, ne=3e-3):
        super(ActorLoss, self).__init__()

        self.ns = ns
        self.nd = nd
        self.ne = ne

        self.anneal = 1e-5

    def forward(self, a, dist_a, V, ve):
        #TODO

        return loss.mean()

class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, V, ve):
        return self.mse(V, ve)
