import numpy as np
from casadi import *
import pickle
import argparse
import os




def compute_mse(C,lambd):
    return dot(C,lambd)



# ---------------------------
# Command-line argument parser
# ---------------------------
parser = argparse.ArgumentParser(description="Run optimization with optional parameters.")

parser.add_argument("--sigma", type=float, default=0.5, help="Label noise std")
parser.add_argument("--a", type=float, default=2.0, help="Parameter a")
parser.add_argument("--b", type=float, default=2.0, help="Parameter b")
parser.add_argument("--N", type=int, default=500, help="Input dimension")
parser.add_argument("--Tmin", type=float, default=2.0, help="Min log10(T)")
parser.add_argument("--Tmax", type=float, default=3.0, help="Max log10(T)")
parser.add_argument("--num_Ts", type=int, default=10, help="Number of T values")
parser.add_argument("--eta_max", type=float, default=2.0, help="Maximum eta")
parser.add_argument("--rho_smooth", type=float, default=1e-4, help="Smoothing regularizer")
parser.add_argument("--avg_m", type=float, default=5, help="Average value of the batch size")
parser.add_argument("--K", type=int, default=100, help="Number of control variables")

parser.add_argument(
    "--batch-opt",
    dest="batch_opt",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Optimize the batch size",
)


args = parser.parse_args()

# ---------------------------
# Use parameters (default or CLI)
# ---------------------------
sigma = args.sigma
a = args.a
b = args.b
N = args.N
Tmin = args.Tmin
Tmax = args.Tmax
num_Ts = args.num_Ts
eta_max = args.eta_max
rho_smooth = args.rho_smooth
batch_opt = args.batch_opt
avg_m = args.avg_m
K = args.K


fin_errors=[]
fin_errors_const=[]
max_eta=[]
opt_etas=[]


for T in np.logspace(Tmin,Tmax,num_Ts):
    
    results = {}

    T=int(T)

    

    ell=T/K
    r=T%K
    
    

    #optimal schedule

    C = MX.sym('C', N)  
    lambd = MX.sym('lambda', N)

    #initial conditions
    C0=np.array([(k+1)**(-a+b) for k in range(N)])
    lambd=np.array([(k+1)**(-b) for k in range(N)])


    #dynamical variables
    C = MX.sym('O',N)
    #control variables
    eta=MX.sym('eta',1)
    m=MX.sym('m',1)

    d = 1 - 2*eta*lambd + ((m+1)/m) * (eta**2) * (lambd*lambd)

    Dc   = d* C  # elementwise multiply
    uTc  = dot(lambd, C) 
    Ac  = Dc + (eta**2) * lambd * uTc /m

    c_next = Ac + (eta**2) * sigma**2 * lambd /m

    step = Function('step', [C, eta, m], [c_next])

    # Start with an empty NLP
    J = 0    #loss function to be optimized
    g=[]     #list of additional (inequality) constraints
    lbg = [] #list of lower bounds of the constraint
    ubg = [] #list of upper bounds of the constraint

    #constrain the average number of active nodes (integrated over the whole timewindow)
    budget=T*avg_m

    # Formulate the NLP
    #map the control problem into a non-linear programming (NLP) problem that we will solve with a ipopt

    #sum variables to impose the budget
    tot=MX(0)

    Ck = MX.sym('CK',N)
    Ck=C0

    
    w = MX.sym('x', 2*K)
    v_eta = w[:K]
    v_m   = w[K:]

    # --- bounds & init ---
    lbw = np.zeros(2*K, dtype=float)
    ubw = np.zeros(2*K, dtype=float)
    w0  = np.zeros(2*K, dtype=float)

    # eta bounds/init
    lbw[:K] = 0.0
    ubw[:K] = eta_max
    if b<a:
        w0 [:K] = 0.01
    else:
        w0 [:K] = 1.

    # m bounds/init
    if batch_opt:
        lower_bound_m = 1.0
        upper_bound_m = 10.0 * avg_m
    else:
        lower_bound_m = avg_m
        upper_bound_m = avg_m

    lbw[K:] = lower_bound_m
    ubw[K:] = upper_bound_m
    w0[K:] = avg_m

    for k in range(T):
        index=int(k/ell)
        #constraint on the control variable
        tot+=v_m[index]
        Ck = step(Ck,v_eta[index],v_m[index])

    #add constraint on the budget
    g+=[budget-tot]
    lbg+=[0]
    ubg+=[0]

    #compute loss - mse on the final performance
    Cf=Ck
    J+=compute_mse(Cf,lambd)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': vertcat(*g)}

    opts = {
    "ipopt": {
        "linear_solver": "mumps",                
        "hessian_approximation": "limited-memory" ,
        "print_level": 4,
        "max_iter": 10000
    },
    "expand": False
    }

    

    # Create an NLP solver
    solver = nlpsol('solver', 'ipopt', prob,opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    #extract optimal trajectories
    eta_opt=[w_opt[i]for i in range(K)]
    m_opt=[w_opt[K+i]for i in range(K)]

    max_eta.append(max(eta_opt))

    results["stats"]=solver.stats()
    
    results["eta_opt"]=eta_opt

    results["m_opt"]=m_opt



    C=C0.copy()
    err=[np.dot(C,lambd)]
    for i in range(T):
        index=int(i/ell)
        C=step(C,eta_opt[index],m_opt[index]).full().flatten().tolist()
        err.append(np.dot(C,lambd))
    fin_errors.append(err[-1])



    results["err_opt"]=err

    #constant eta=1 schedule
    C=C0.copy()
    err=[np.dot(C,lambd)]
    for i in range(T):
        C=step(C,1,avg_m).full().flatten().tolist()
        err.append(np.dot(C,lambd))
    fin_errors.append(err[-1])

    results["err_eta1"]=err


    #optimal constant-eta schedule

    # Start with an empty NLP
    w=[] #list of variables to be optimized - including both dynamical and control variables
    w0 = [] #list of initial conditions
    lbw = [] #list of lower bounds
    ubw = [] #list of upper bounds
    J = 0    #loss function to be optimized
    g=[]     #list of additional (inequality) constraints
    lbg = [] #list of lower bounds of the constraint
    ubg = [] #list of upper bounds of the constraint

    #constrain the average number of active nodes (integrated over the whole timewindow)
    budget=T*avg_m

    # Formulate the NLP
    #map the control problem into a non-linear programming (NLP) problem that we will solve with a ipopt

    #sum variables to impose the budget
    tot=MX(0)

    Ck = MX.sym('CK',N)
    Ck=C0

    k=0

    eta_variable = MX.sym('eta',1)

    w += [eta_variable]
    lbw += [0 ]
    ubw +=[ eta_max]
    w0 += [0.1]

    for k in range(T):
        Ck = step(Ck,eta_variable,avg_m)


    #compute loss - mse on the final performance
    Cf=Ck
    J+=compute_mse(Cf,lambd)

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    
    
    # Create an NLP solver
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    results["stats_const_eta"]=solver.stats()

    
    #extract optimal trajectories
    eta_opt_const=w_opt

    results["eta_opt_const"]=eta_opt_const

    opt_etas.append(eta_opt_const)

    C=C0.copy()
    err=[np.dot(C,lambd)]
    for i in range(T):
        C=step(C,eta_opt_const,avg_m).full().flatten().tolist()
        err.append(np.dot(C,lambd))
    fin_errors_const.append(err[-1])

    results["err_opt_const"]=err


    #optimal power-law schedule

    # Start with an empty NLP
    w=[] #list of variables to be optimized - including both dynamical and control variables
    w0 = [] #list of initial conditions
    lbw = [] #list of lower bounds
    ubw = [] #list of upper bounds
    J = 0    #loss function to be optimized
    g=[]     #list of additional (inequality) constraints
    lbg = [] #list of lower bounds of the constraint
    ubg = [] #list of upper bounds of the constraint

    #constrain the average number of active nodes (integrated over the whole timewindow)
    budget=T*avg_m

    # Formulate the NLP
    #map the control problem into a non-linear programming (NLP) problem that we will solve with a ipopt

    #sum variables to impose the budget

    Ck = MX.sym('CK',N)
    Ck=C0

    k=0

    exponent = MX.sym('exponent',1)
    coefficient = MX.sym('coefficient',1)

    w += [exponent]
    lbw += [0 ]
    ubw +=[inf]
    w0 += [1]

    w += [coefficient]
    lbw += [0 ]
    ubw +=[eta_max]
    w0 += [0.1]

    for k in range(T):
        Ck = step(Ck,coefficient*DM(k+1)**(-exponent),avg_m)

    #compute loss - mse on the final performance
    Cf=Ck
    J+=compute_mse(Cf,lambd)

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}

    


    # Create an NLP solver
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    results["stats_power_law_eta"]=solver.stats()
    #extract optimal trajectories
    opt_exp=w_opt[0]
    opt_coeff=w_opt[1]

    results["opt_exp"]=opt_exp

    results["opt_coeff"]=opt_coeff


    C=C0.copy()
    err_opt_power_law=[np.dot(C,lambd)]
    for i in range(T):
        C=step(C,opt_coeff*(i+1)**(-opt_exp),avg_m).full().flatten().tolist()
        err_opt_power_law.append(np.dot(C,lambd))
    results["err_opt_power_law"]=err_opt_power_law

    filename = f"data/results_N_{N}_T_{T}_sigma_{sigma}_a_{a}_b_{b}_batch_{batch_opt}_avgm_{avg_m}_K_{K}_rho_{rho_smooth}_etamax_{eta_max}.pkl"

    # Save
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(results, f)

