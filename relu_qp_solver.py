
import torch
import numpy as np
import pandas as pd
from relu_qp_data import Settings, Results, Info, QP, is_use_cpu, mpc_horizon
import timeit
import time

class ReLU_Layer(torch.nn.Module):
    def __init__(self, QP=None, settings=Settings()):
        super(ReLU_Layer, self).__init__()

        torch.set_default_dtype(settings.precision)
        self.QP = QP
        self.settings = settings
        self.rhos = self.setup_rhos()
        
        self.W_ks, self.B_ks, self.b_ks = self.setup_matrices()
        self.clamp_inds = (self.QP.nx, self.QP.nx + self.QP.nc)

    def setup_rhos(self):
        """
        Setup rho values for ADMM
        """
        stng = self.settings
        rhos = [stng.rho]
        if stng.adaptive_rho:
            rho = stng.rho/stng.adaptive_rho_tolerance
            while rho >= stng.rho_min:
                rhos.append(rho)
                rho = rho/stng.adaptive_rho_tolerance
            rho = stng.rho*stng.adaptive_rho_tolerance
            while rho <= stng.rho_max:
                rhos.append(rho)
                rho = rho*stng.adaptive_rho_tolerance
            rhos.sort()
        # conver to torch tensor
        rhos = torch.tensor(rhos, device=stng.device, dtype=stng.precision).contiguous()
        return rhos
    
    def setup_matrices(self):
        """
        Setup ADMM matrices for ReLU-QP solver for each rho
        """
        # unpack values
        H, g, A, l, u = self.QP.H, self.QP.g, self.QP.A, self.QP.l, self.QP.u
        nx, nc = self.QP.nx, self.QP.nc
        sigma = self.settings.sigma
        stng = self.settings

        # Calculate kkt_rhs_invs
        kkt_rhs_invs = []
        for rho_scalar in self.rhos:
            rho = rho_scalar * torch.ones(nc).to(g)
            rho[(u - l) <= stng.eq_tol] = rho_scalar * 1e3
            rho = torch.diag(rho)
            kkt_rhs_invs.append(torch.inverse(H + sigma * torch.eye(nx).to(g) + A.T @ (rho @ A)))

        W_ks = {}
        B_ks = {}
        b_ks = {}
        
        # Other layer updates for each rho
        for rho_ind, rho_scalar in enumerate(self.rhos):
            rho = rho_scalar * torch.ones(nc, device=stng.device, dtype=stng.precision).contiguous()
            rho[(u - l) <= stng.eq_tol] = rho_scalar * 1e3
            rho_inv = torch.diag(1.0 / rho)
            rho = torch.diag(rho).to(device=stng.device, dtype=stng.precision).contiguous()
            K = kkt_rhs_invs[rho_ind]
            Ix = torch.eye(nx, device=stng.device, dtype=stng.precision).contiguous()
            Ic = torch.eye(nc, device=stng.device, dtype=stng.precision).contiguous()
            W_ks[rho_ind] = torch.cat([
                torch.cat([ K @ (sigma * Ix - A.T @ (rho @ A)),           2 * K @ A.T @ rho,            -K @ A.T], dim=1),
                torch.cat([ A @ K @ (sigma * Ix - A.T @ (rho @ A)) + A,   2 * A @ K @ A.T @ rho - Ic,  -A @ K @ A.T + rho_inv], dim=1),
                torch.cat([ rho @ A,                                      -rho,                         Ic], dim=1)
            ], dim=0).contiguous()
            B_ks[rho_ind] = torch.cat([-K, -A @ K, torch.zeros(nc, nx).to(g)], dim=0).contiguous()
            b_ks[rho_ind] = (B_ks[rho_ind] @ g).contiguous()
        return W_ks, B_ks, b_ks

    def forward(self, input, idx):
        input = self.jit_forward(input, self.W_ks[idx], self.b_ks[idx], self.QP.l, self.QP.u, self.clamp_inds[0], self.clamp_inds[1])
        #print(self.clamp_inds[0], self.clamp_inds[1])
        return input
    
    @torch.jit.script
    def jit_forward(input, W, b, l, u, idx1: int, idx2: int):
        torch.matmul(W, input, out=input)
        input.add_(b)
        input[idx1:idx2].clamp_(l, u)
        return input
    

class ReLU_QP(object):
    def __init__(self):
        super().__init__()

        self.info = Info()
        self.results = Results(info=self.info)

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def setup(self, H, g, A, l, u, 
                        verbose=False,
                        warm_starting=True,
                        scaling=False, #todo: implement scaling
                        rho=1e-3,
                        rho_min=1e-9,
                        rho_max=1e3,
                        sigma=1e-6,
                        adaptive_rho=True,
                        adaptive_rho_interval=1,
                        adaptive_rho_tolerance=10,
                        max_iter=1000,
                        eps_abs=1e-3,
                        check_interval=4,
                        device=torch.device("cuda" if torch.cuda.is_available() and not is_use_cpu else "cpu"), 
                        precision= torch.float64):
        """
        Setup ReLU-QP solver problem of the form

        minimize     1/2 x' * H * x + g' * x
        subject to   l <= A * x <= u

        solver settings can be specified as additional keyword arguments
        """
        self.start.record()

        self.settings = Settings(verbose=verbose,
                                    warm_starting=warm_starting,
                                    scaling=scaling,
                                    rho=rho,
                                    rho_min=rho_min,
                                    rho_max=rho_max,
                                    sigma=sigma,
                                    adaptive_rho=adaptive_rho,
                                    adaptive_rho_interval=adaptive_rho_interval,
                                    adaptive_rho_tolerance=adaptive_rho_tolerance,
                                    max_iter=max_iter,
                                    eps_abs=eps_abs,
                                    check_interval=check_interval,
                                    device=device,
                                    precision=precision)

        self.QP = QP(H, g, A, l, u)

        self.layers = ReLU_Layer(QP=self.QP, settings=self.settings)
        
        self.x = torch.zeros(self.QP.nx).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.z = torch.zeros(self.QP.nc).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.lam = torch.zeros(self.QP.nc).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.output = torch.cat([self.x, self.z, self.lam]).to(device=self.settings.device, dtype=self.settings.precision).contiguous()

        self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - self.settings.rho))

        self.end.record()
        torch.cuda.synchronize()
        self.results.info.setup_time = self.start.elapsed_time(self.end)

    def update(self, g=None, l=None, u=None,
               Hx=None, Ax=None):
        """
        Update ReLU-QP problem arguments
        """
        self.start.record()
        # todo update vectors
        if g is not None:
            self.QP.g = torch.from_numpy(g).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
            for (i, rho) in enumerate(self.layers.rhos):
                self.layers.b_ks[i] = self.layers.B_ks[i] @ self.QP.g
            
        if l is not None:
            self.QP.l = torch.from_numpy(l).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        if u is not None:
            self.QP.u = torch.from_numpy(u).to(device=self.settings.device, dtype=self.settings.precision).contiguous()

        # assert that matrices cannot be changed for now
        assert Hx is None and Ax is None, "updating Hx and Ax is not supported yet"
        
        self.end.record()
        torch.cuda.synchronize()
        self.results.info.update_time = self.start.elapsed_time(self.end)

        return None
    
    def update_settings(self, **kwargs):
        """
        Update ReLU-QP solver settings

        It is possible to change: 'max_iter', 'eps_abs', 
                                  'verbose', 
                                  'check_interval',
        """
        for key, value in kwargs.items():
            if key in ["max_iter", "eps_ab", "verbose", "check_interval"]:
                setattr(self.settings, key, value)
            elif key in ["rho", "rho_min", "rho_max", "sigma", "adaptive_rho", "adaptive_rho_interval", "adaptive_rho_tolerance"]:
                raise ValueError("Cannot change {} after setup".format(key))
            else:
                raise ValueError("Invalid setting: {}".format(key))

    def solve(self):
        """
        Solve QP Problem
        """
        self.start.record()

        stng = self.settings
        nx, nc = self.QP.nx, self.QP.nc

        rho = self.layers.rhos[self.rho_ind]

        for k in range(1, stng.max_iter + 1):
            self.output = self.layers(self.output, self.rho_ind)
            # rho update
            if k % stng.check_interval == 0 and stng.adaptive_rho:
                self.x, self.z, self.lam = self.output[:nx], self.output[nx:nx+nc], self.output[nx+nc:nx+2*nc]
                primal_res, dual_res, rho = self.compute_residuals(self.QP.H, 
                        self.QP.A, self.QP.g, self.x, self.z, self.lam, rho, stng.rho_min, stng.rho_max)

                if rho > self.layers.rhos[self.rho_ind] * stng.adaptive_rho_tolerance and self.rho_ind < len(self.layers.rhos) - 1:
                    self.rho_ind += 1
                    self.rho_ind = min(self.rho_ind, len(self.layers.rhos) - 1)
                elif rho < self.layers.rhos[self.rho_ind] / stng.adaptive_rho_tolerance and self.rho_ind > 0:
                    self.rho_ind -= 1
                
                if stng.verbose:
                    print('Iter: {}, rho: {:.2e}, res_p: {:.2e}, res_d: {:.2e}'.format(k, rho, primal_res, dual_res))

                # check convergence
                if primal_res < stng.eps_abs * np.sqrt(nc) and dual_res < stng.eps_abs * np.sqrt(nx):

                    self.update_results(iter=k,
                                        status="solved",
                                        pri_res=primal_res,
                                        dua_res=dual_res,
                                        rho_estimate=rho)
                    
                    return self.results

        primal_res, dual_res, rho = self.compute_residuals(self.QP.H, self.QP.A, self.QP.g, self.x, self.z, self.lam, rho, stng.rho_min, stng.rho_max)
        self.update_results(iter=stng.max_iter, 
                            status="max_iters_reached", 
                            pri_res=primal_res, 
                            dua_res=dual_res, 
                            rho_estimate=rho)
        return self.results

    def warm_start(self, x:torch.tensor or np.ndarray = None, 
                   z:torch.tensor or np.ndarray = None, 
                   lam:torch.tensor or np.ndarray = None, 
                   rho:float = None):
        """
        Warm start primal or dual variables, lagrange multipliers, and rho
        """
        if x is not None:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            self.x = x.to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        
        if z is not None:
            if isinstance(z, np.ndarray):
                z = torch.from_numpy(z)
            self.z = z.to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        
        if lam is not None:
            if isinstance(lam, np.ndarray):
                lam = torch.from_numpy(lam)
            self.lam = lam.to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        
        if rho is not None:
            self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - rho))

        return None

    def update_results(self, iter=None, 
                       status=None, 
                       pri_res=None, 
                       dua_res=None, 
                       rho_estimate=None):
        """
        Update results and info
        """

        self.results.x = self.x
        self.results.z = self.z

        self.results.info.iter = iter
        self.results.info.status = status
        self.results.info.obj_val = self.compute_Cost(H=self.QP.H, g=self.QP.g, x=self.x)
        self.results.info.pri_res = pri_res
        self.results.info.dua_res = dua_res
        self.results.info.rho_estimate = rho_estimate
        self.info.update_time = 0 #todo: implement in update method
        self.end.record()
        torch.cuda.synchronize()
        run_time = self.start.elapsed_time(self.end) #unit: ms
        
        self.results.info.run_time = run_time
        self.results.info.solve_time = self.results.info.update_time + run_time
        self.lam = torch.zeros(self.QP.nc).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        if not self.settings.warm_starting:
            self.clear_primal_dual()
    
    @torch.jit.script
    def compute_residuals(H, A, g, x, z, lam, rho, rho_min: float, rho_max: float):
        t1 = torch.matmul(A, x)
        t2 = torch.matmul(H, x)
        t3 = torch.matmul(A.T, lam)

        primal_res = torch.linalg.vector_norm(t1 - z, ord=torch.inf)
        dual_res = torch.linalg.vector_norm(t2 + t3 + g, ord=torch.inf)
        numerator = torch.div(primal_res, torch.max(torch.linalg.vector_norm(t1, ord=torch.inf), torch.linalg.vector_norm(z, ord=torch.inf)))
        denom = torch.div(dual_res, torch.max(torch.max(torch.linalg.vector_norm(t2, ord=torch.inf), torch.linalg.vector_norm(t3, ord=torch.inf)), torch.linalg.vector_norm(g, ord=torch.inf)))
        rho = torch.clamp(rho * torch.sqrt(numerator / denom), rho_min, rho_max)
        return primal_res, dual_res, rho
    
    @torch.jit.script
    def compute_Cost(H=None, g=None, x=None):
        return 0.5*torch.dot(x,torch.matmul(H,x)) + torch.dot(g,x)

    def clear_primal_dual(self):
        """
        Clear primal and dual variables and reset rho index
        """
        self.x = torch.zeros(self.QP.nx).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.z = torch.zeros(self.QP.nc).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.lam = torch.zeros(self.QP.nc).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.output = torch.cat([self.x, self.z, self.lam]).to(device=self.settings.device, dtype=self.settings.precision).contiguous()
        self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - self.settings.rho))
        return None
    
    # todo: implement scaling
    # todo: better verbose printing
    
if __name__ == "__main__":
    # test on simple QP
    # min 1/2 x' * H * x + g' * x
    # s.t. l <= A * x <= u
    # H = torch.tensor([[6, 2, 1], [2, 5, 2], [1, 2, 4.0]], dtype=torch.double)
    # g = torch.tensor([-8.0, -3, -3], dtype=torch.double)
    # A = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
    # l = torch.tensor([3.0, 0, -10.0, -10, -10], dtype=torch.double)
    # u = torch.tensor([3.0, 0, torch.inf, torch.inf, torch.inf], dtype=torch.double)
    H = pd.read_csv('H_txt.csv', header = None)
    H = torch.tensor(H.values).view(12 * mpc_horizon, 12 * mpc_horizon)
    g = pd.read_csv('g_txt.csv', header = None)
    g = torch.tensor(g.values).view(12 * mpc_horizon)
    A = pd.read_csv('A_txt.csv', header = None)
    A = torch.tensor(A.values).view(16 * mpc_horizon + 12, 12 * mpc_horizon)
    l = pd.read_csv('l_txt.csv', header = None)
    l = torch.tensor(l.values).view(16 * mpc_horizon + 12)
    u = pd.read_csv('u_txt.csv', header = None)
    u = torch.tensor(u.values).view(16 * mpc_horizon + 12)

    qp = ReLU_QP()
    qp.setup(H=H, g=g, A=A, l=l, u=u)
    results = qp.solve()

    print("iter num: ", results.info.iter)
    print("solve status: ", results.info.status)
    print("opt result: ")
    for i in range(10):
            print(results.x[i * 12: (i + 1) * 12])

    print("setup time(ms): ", results.info.setup_time)
    print("slove time(ms): ", results.info.solve_time)


