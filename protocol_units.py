import numba as nb
import numpy as np


__all__ = [
    "depolarizing_noise", "dephasing_noise", "amplitude_damping", "bit_phase_flip",
    "get_swap_prob_suc", "get_swap_lambda_out",
    "get_dist_lambda_out", "get_dist_prob_fail", "get_dist_prob_suc",
    "memory_cut_off", "fidelity_cut_off", "run_time_cut_off", "time_cut_off",
]


"""
This module contain the defined success probability and
resulting output parameters of each protocol unit.
"""
########################################################################
"""
Error model 

"""
@nb.jit(nopython=True, error_model="numpy")
def depolarizing_noise(lambdas, t, p):
    """Applies depolarizing noise to the Bell Diagonal state, ensuring normalization."""
    p_t = 1 - np.exp(-p * t)
    # Apply depolarization to each lambda
    new_lambdas = [(1 - p_t) * lambda_0 + p_t / 4 for lambda_0 in lambdas]
    new_lambdas = np.asarray(new_lambdas)
    return new_lambdas

@nb.jit(nopython=True, error_model="numpy")
def dephasing_noise(lambdas, t, gamma):
    """Applies dephasing noise (affecting lambda_2 and lambda_3) with normalization."""
    decay_factor = (1 - np.exp(-gamma * t))/2
    new_lambdas = [
        lambdas[0] * (1 - decay_factor) + lambdas[1] * decay_factor,  # Redistribute lost probability
        lambdas[1] * (1 - decay_factor) + lambdas[0] * decay_factor,
        lambdas[2] * (1 - decay_factor) + lambdas[3] * decay_factor,
        lambdas[3] * (1 - decay_factor) + lambdas[2] * decay_factor   # Redistribute lost probability
    ]
    new_lambdas = np.asarray(new_lambdas)
    return new_lambdas

@nb.jit(nopython=True, error_model="numpy")
def amplitude_damping(lambdas, t, gamma):
    """Models amplitude damping noise affecting lambda_1 and lambda_4, ensuring probability conservation."""
    p_t = 1 - np.exp(-gamma * t)
    lost_probability = (lambdas[0] - lambdas[3]) * p_t
    new_lambdas = [
        lambdas[0] - lost_probability,
        lambdas[1],
        lambdas[2],
        lambdas[3] + lost_probability
    ]
    new_lambdas = np.asarray(new_lambdas)
    return new_lambdas

@nb.jit(nopython=True, error_model="numpy")
def bit_phase_flip(lambdas, t, p):
    """Applies bit-flip or phase-flip errors (affecting lambda_2 and lambda_3) with normalization."""
    factor = (1 - 2 * p * (1 - np.exp(-t)))
    lost_probability = (1 - factor) * (lambdas[1] + lambdas[2])
    new_lambdas = [
        lambdas[0] + lost_probability / 2,  # Redistribute lost probability
        lambdas[1] * factor,
        lambdas[2] * factor,
        lambdas[3] + lost_probability / 2   # Redistribute lost probability
    ]
    new_lambdas = np.asarray(new_lambdas)
    return new_lambdas

"""
Success probability p and
the resulting output parameters of swap and distillation.

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
lambdas1, lambdas2 : float
    The parameters of the two input links.
depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate :
    Error parameters

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def get_one(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get a trivial one
    """
    return np.ones(len(link1))


@nb.jit(nopython=True, error_model="numpy")
def get_swap_prob_suc(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get w_swap
    """
    output =  (link1[0] * link2[0] + link1[4] * link2[1] - link1[8] * link2[2] + link1[12] * link2[3]) / 2 

    return [output, output, output, output]


@nb.jit(nopython=True, error_model="numpy")
def get_swap_lambda_out(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get w_swap
    """
    get_swap_prob = get_swap_prob_suc(t1, t2, link1, link2, depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate) 
    link = np.zeros(len(link1))
    link[0] = ((link1[0] + link1[3]) * (link2[0] + link2[12]) + (link1[4] + link1[7]) * (link2[1] + link2[13]) - (link1[8] + link1[11]) * (link2[2] + link2[14]) + (link1[12] + link1[15]) * (link2[3] + link2[15])) / (8 * get_swap_prob[0])
    link[1] = ((link1[1] + link1[2]*1j) * (link2[0] + link2[12]) + (link1[5] + link1[6]*1j) * (link2[1] + link2[13]) - (link1[9] + link1[10]*1j) * (link2[2] + link2[14]) + (link1[13] + link1[14]*1j) * (link2[3] + link2[15])) / (8 * get_swap_prob[1])
    link[2] = ((link1[0] + link1[3]) * (link2[4] + link2[8]*1j) + (link1[4] + link1[7]) * (link2[5] + link2[9]*1j) - (link1[8] + link1[11]) * (link2[6] + link2[10]*1j) + (link1[12] + link1[14]) * (link2[7] + link2[11]*1j)) / (8 * get_swap_prob[2])
    link[3] = ((link1[1] + link1[2]*1j) * (link2[4] + link2[8]*1j) + (link1[5] + link1[6]*1j) * (link2[5] + link2[9]*1j) - (link1[9] + link1[10]*1j) * (link2[6] + link2[10]*1j) + (link1[13] + link1[14]*1j) * (link2[7] + link2[11]*1j)) / (8 * get_swap_prob[3])   
    link[4] = ((link1[1] - link1[2]*1j) * (link2[0] + link2[12]) + (link1[5] - link1[6]*1j) * (link2[1] + link2[13]) - (link1[9] - link1[10]*1j) * (link2[2] + link2[14]) + (link1[13] - link1[14]*1j) * (link2[3] + link2[15])) / (8 * get_swap_prob[4])   
    link[5] = ((link1[0] - link1[3]) * (link2[0] + link2[12]) + (link1[4] - link1[7]) * (link2[1] + link2[13]) - (link1[8] - link1[11]) * (link2[2] + link2[14]) + (link1[12] - link1[15]) * (link2[3] + link2[15])) / (8 * get_swap_prob[5])
    link[6] = ((link1[1] - link1[2]*1j) * (link2[4] + link2[8]*1j) + (link1[5] - link1[6]*1j) * (link2[5] + link2[9]*1j) - (link1[9] - link1[10]*1j) * (link2[6] + link2[10]*1j) + (link1[13] - link1[14]*1j) * (link2[7] + link2[11]*1j)) / (8 * get_swap_prob[6])
    link[7] = ((link1[0] - link1[3]) * (link2[4] + link2[8]*1j) + (link1[4] - link1[7]) * (link2[5] + link2[9]*1j) - (link1[8] - link1[11]) * (link2[6] + link2[10]*1j) + (link1[12] - link1[14]) * (link2[7] + link2[11]*1j)) / (8 * get_swap_prob[7])
    link[8] = ((link1[0] + link1[3]) * (link2[4] - link2[8]*1j) + (link1[4] + link1[7]) * (link2[5] - link2[9]*1j) - (link1[8] + link1[11]) * (link2[6] - link2[10]*1j) + (link1[12] + link1[14]) * (link2[7] - link2[11]*1j)) / (8 * get_swap_prob[8])
    link[9] = ((link1[1] + link1[2]*1j) * (link2[4] - link2[8]*1j) + (link1[5] + link1[6]*1j) * (link2[5] - link2[9]*1j) - (link1[9] + link1[10]*1j) * (link2[6] - link2[10]*1j) + (link1[13] + link1[14]*1j) * (link2[7] - link2[11]*1j)) / (8 * get_swap_prob[9])
    link[10] = ((link1[0] + link1[3]) * (link2[0] - link2[12]) + (link1[4] + link1[7]) * (link2[1] - link2[13]) - (link1[8] + link1[11]) * (link2[2] - link2[14]) + (link1[12] + link1[15]) * (link2[3] - link2[15])) / (8 * get_swap_prob[10])
    link[11] = ((link1[1] + link1[2]*1j) * (link2[0] - link2[12]) + (link1[5] + link1[6]*1j) * (link2[1] - link2[13]) - (link1[9] + link1[10]*1j) * (link2[2] - link2[14]) + (link1[13] + link1[14]*1j) * (link2[3] - link2[15])) / (8 * get_swap_prob[11])
    link[12] = ((link1[1] - link1[2]*1j) * (link2[4] - link2[8]*1j) + (link1[5] - link1[6]*1j) * (link2[5] - link2[9]*1j) - (link1[9] - link1[10]*1j) * (link2[6] - link2[10]*1j) + (link1[13] - link1[14]*1j) * (link2[7] - link2[11]*1j)) / (8 * get_swap_prob[12])
    link[13] = ((link1[0] - link1[3]) * (link2[4] - link2[8]*1j) + (link1[4] + link1[7]) * (link2[5] - link2[9]*1j) - (link1[8] + link1[11]) * (link2[6] - link2[10]*1j) + (link1[12] + link1[14]) * (link2[7] - link2[11]*1j)) / (8 * get_swap_prob[13])
    link[14] = ((link1[1] - link1[2]*1j) * (link2[0] - link2[12]) + (link1[5] - link1[6]*1j) * (link2[1] - link2[13]) - (link1[9] - link1[10]*1j) * (link2[2] - link2[14]) + (link1[13] - link1[14]*1j) * (link2[3] - link2[15])) / (8 * get_swap_prob[14])
    link[15] = ((link1[0] - link1[3]) * (link2[0] - link2[12]) + (link1[4] - link1[7]) * (link2[1] - link2[13]) - (link1[8] - link1[11]) * (link2[2] - link2[14]) + (link1[12] - link1[15]) * (link2[3] - link2[15])) / (8 * get_swap_prob[15]) 
    link = depolarizing_noise(link, np.abs(t1-t2), depolar_rate)
    link = dephasing_noise(link, np.abs(t1-t2), dephase_rate)

    if sum(link) > 1.1 or sum(link) < 0.9:
        print(link)
        print(sum(link))
        raise ValueError(f"sum(lambdas) > 1")
    return link

@nb.jit(nopython=True, error_model="numpy")
def get_dist_lambda_out(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get p_dist * w_dist
    """
    if t1 < t2:
        link1 = depolarizing_noise(link1, np.abs(t1-t2), depolar_rate)
        link1 = dephasing_noise(link1, np.abs(t1-t2), dephase_rate)
    else:
        link2 = depolarizing_noise(link2, np.abs(t1-t2), depolar_rate)
        link2 = dephasing_noise(link2, np.abs(t1-t2), dephase_rate)
        
    
    link = np.zeros(len(link1))
    get_dist_prob = get_dist_prob_suc(t1, t2, link1, link2, depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate)
    link[0] = (link1[0] + link1[15]) * (link2[0] + link2[3] + link2[12] + link2[15]) / (8 * get_dist_prob[0])
    link[1] = (link1[1] + link1[14] * 1j) * (link2[1] + link2[2]*1j + link2[13] + link2[14]*1j) / (8 * get_dist_prob[1])
    link[2] = (link1[4] + link1[11] * 1j) * (link2[4] + link2[7] + link2[8]*1j + link2[11]*1j) / (8 * get_dist_prob[2])
    link[3] = (link1[5] - link1[10]) * (link2[5] + link2[6]*1j + link2[9]*1j - link2[10]) / (8 * get_dist_prob[3])
    link[4] = (link1[1] - link1[14] * 1j) * (link2[1] - link2[2]*1j + link2[13] - link2[14]*1j) / (8 * get_dist_prob[4])
    link[5] = (link1[0] - link1[15]) * (link2[0] - link2[3] + link2[12] - link2[15]) / (8 * get_dist_prob[5])
    link[6] = (link1[5] + link1[10]) * (link2[5] - link2[6]*1j + link2[9]*1j + link2[10]) / (8 * get_dist_prob[6])
    link[7] = (link1[4] - link1[11] * 1j) * (link2[4] - link2[7] + link2[8]*1j - link2[11]*1j) / (8 * get_dist_prob[7])
    link[8] = (link1[4] - link1[11] * 1j) * (link2[4] + link2[7] - link2[8]*1j - link2[11]*1j) / (8 * get_dist_prob[8])
    link[9] = (link1[5] + link1[10]) * (link2[5] + link2[6]*1j - link2[9]*1j + link2[10]) / (8 * get_dist_prob[9])
    link[10] = (link1[0] - link1[15]) * (link2[0] + link2[3] - link2[12] - link2[15]) / (8 * get_dist_prob[10])
    link[11] = (link1[1] - link1[14] * 1j) * (link2[1] + link2[2]*1j - link2[13] - link2[14]*1j) / (8 * get_dist_prob[11])
    link[12] = (-link1[5] + link1[10] * 1j) * (-1*link2[5] + link2[6]*1j + link2[9]*1j - link2[10]) / (8 * get_dist_prob[12])
    link[13] = (link1[4] + link1[11] * 1j) * (link2[4] - link2[7] - link2[8]*1j + link2[11]*1j) / (8 * get_dist_prob[13])
    link[14] = (link1[1] + link1[14] * 1j) * (link2[1] - link2[2]*1j - link2[13] + link2[14]*1j) / (8 * get_dist_prob[14])
    link[15] = (link1[0] + link1[15]) * (link2[0] - link2[3] - link2[12] + link2[15]) / (8 * get_dist_prob[15])
    
    if sum(link) > 1.1 or sum(link) < 0.9:
        print(link)
        print(sum(link))
        raise ValueError(f"sum(lambdas) > 1")
    return link

@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_fail(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get 1 - p_dist
    """
    get_dist_prob = get_dist_prob_suc(t1, t2, link1, link2, depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate)  
    one = np.ones(len(link1))
    output = [x-y for x, y in zip(one, get_dist_prob)]
    return output


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_suc(t1, t2, link1, link2, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Get p_dist
    """

    output = (link1[0] * link2[0] + link1[15] * link2[15])  / 2
    return [output, output, output, output]







########################################################################
"""
Cut-off functions

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
lambdas1, lambdas2: float
    The Bell diagonal coefficients of the two input links.
mt_cut: int
    The memory time cut-off.
f_cut: float
    The fidelity parameter cut-off. (0 < lambda[0] < 1)
    Set a cut-off on the input links's bell diagonal coefficient
t_coh: int or float
    The memory coherence time.

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def memory_cut_off(
        t1, t2, lambdas1=[1.0, 0., 0., 0.], lambdas2=[1.0, 0., 0., 0.],
        mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max):
    """
    Memory storage cut-off. The two input links suvives only if
    |t1-t2|<=mt_cut
    """
    if abs(t1 - t2) > mt_cut:
        # constant shift mt_cut is added in the iterative convolution
        return min(t1, t2), False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def fidelity_cut_off(
    t1, t2, lambdas1=[1.0, 0., 0., 0.], lambdas2=[1.0, 0., 0., 0.],
    mt_cut=np.iinfo(int).max, f_cut=1.e-8, rt_cut=np.iinfo(int).max):
    """
    Fidelity-dependent cut-off, The two input links suvives only if
    lambdas1 <= f_cut and lambdas2 <= f_cut including decoherence.
    """

    if t1 == t2:
        if lambdas1[0] < f_cut or lambdas2[0] < f_cut:
            return t1, False
        return t1, True
    if t1 > t2:  # make sure t1 < t2
        t1, t2 = t2, t1
        lambdas1, lambdas2 = lambdas2, lambdas1
    # first link has low quality
    if lambdas1[0] < f_cut:
        return t1, False  # waiting_time = min(t1, t2)
    waiting = int(np.floor(np.log(lambdas1[0]/f_cut)))
    # first link waits too long
    if t1 + waiting < t2:
        return t1 + waiting, False  # min(t1, t2) < waiting_time < max(t1, t2)
    # second link has low quality
    elif lambdas2[0] < f_cut:
        return t2, False  # waiting_time = max(t1, t2)
    # both links are good
    else:
        return t2, True  # waiting_time = max(t1, t2)


@nb.jit(nopython=True, error_model="numpy")
def run_time_cut_off(
    t1, t2, lambdas1, lambdas2,
    mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max):
    if t1 > rt_cut or t2 > rt_cut:
        return rt_cut, False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def time_cut_off(
    t1, t2, lambdas1, lambdas2,
    mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max, ):
    waiting_time1, result1 = memory_cut_off(
        t1, t2, lambdas1, lambdas2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut)
    waiting_time2, result2 = run_time_cut_off(
        t1, t2, lambdas1, lambdas2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut)
    result1 += mt_cut
    result2 += rt_cut
    if result1 and result2:
        return max(waiting_time1, waiting_time2), True
    else:
        # the waiting time of failing cutoff is always
        # smaller than max(t1, t2), so we just need a min here.
        return min(waiting_time1, waiting_time2), False


########################################################################
def join_links_compatible(
        pmf1, pmf2, lambda_func1, lambda_func2, ycut=True,
        cutoff=np.iinfo(int).max, 
        cut_type="memory_time", evaluate_func=get_one, 
        depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
    """
    Calculate P_s and P_f.
    Calculate sum_(t=tA+tB) Pr(TA=tA)*Pr(TB=tB)*f(tA, tB)
    where f is the value function to
    be evaluated for the joint distribution.

    Note
    ----
    For swap the success probability p is
    considered in the iterative convolution.

    For the memory time cut-off,
    the constant shift is added in the iterative convolution.


    Parameters
    ----------
    pmf1, pmf2: array-like
        The waiting time distribution of the two input links, Pr(T=t).
    lambda_func1, lambda_func2: array-like
        The Bell diagonal coefficient function, W(t).
    cutoff: int or float
        The cut-off threshold.
    ycut: bool
        Successful cut-off or failed cut-off.
    cutoff_type: str
        Type of cut-off.
        `memory_time`, `run_time` or `fidelity`.
    evaluate_func: str
        The function to be evaluated the returns a float number.
        It can be
        ``get_one`` for trival cases\n
        ``get_swap_lambda_out`` for lambdaswap\n
        ``get_dist_prob_suc`` for pdist\n
        ``get_dist_prob_fail`` for 1-pdist\n
        ``get_dist_lambda_out`` for pdist * lambdadist
    t_coh: int or float
        The coherence time of the memory.

    Returns
    -------
    result: array-like 1-D
        The resulting array of joining the two links.
    """
    mt_cut=np.iinfo(int).max
    w_cut=0.0
    rt_cut=np.iinfo(int).max
    twod_par = False
    if cut_type == "memory_time":
        cutoff_func = memory_cut_off
        mt_cut = cutoff
    elif cut_type == "fidelity":
        cutoff_func = fidelity_cut_off
        w_cut = cutoff
    elif cut_type == "run_time":
        cutoff_func = run_time_cut_off
        rt_cut = cutoff
    else:
        raise NotImplementedError("Unknow cut-off type")

    if evaluate_func == "1":
        evaluate_func = get_one
    elif evaluate_func == "get_swap_prob_suc":
        evaluate_func = get_swap_prob_suc
    elif evaluate_func == "w1w2":
        evaluate_func = get_swap_lambda_out
    elif evaluate_func == "0.5+0.5w1w2":
        evaluate_func = get_dist_prob_suc
    elif evaluate_func == "0.5-0.5w1w2":
        evaluate_func = get_dist_prob_fail
    elif evaluate_func == "w1+w2+4w1w2":
        evaluate_func = get_dist_lambda_out
    elif isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    
    result = join_links_helper(
        pmf1, pmf2, lambda_func1, lambda_func2, cutoff_func=cutoff_func, evaluate_func=evaluate_func, ycut=ycut, 
        mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
    return result


@nb.jit(nopython=True, error_model="numpy")
def join_links_helper(
        pmf1, pmf2, w_func1, w_func2,
        cutoff_func=memory_cut_off, evaluate_func=get_one, ycut=True, mt_cut=np.iinfo(int).max, w_cut=0.0, rt_cut=np.iinfo(int).max, 
        depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):   
    size = len(pmf1)
    result = np.zeros((size, 16), dtype=np.float64)  # Modify result to be a 2D array
    for t1 in range(1, size):
        for t2 in range(1, size):
            waiting_time, selection_pass = cutoff_func(
                t1, t2, w_func1[t1], w_func2[t2],
                mt_cut, w_cut, rt_cut)
            if not ycut:
                selection_pass = not selection_pass
            if selection_pass:
                output = evaluate_func(t1, t2, w_func1[t1], w_func2[t2], depolar_rate, dephase_rate, amplitude_damping_rate, bit_phase_flip_rate)
                
                for i in range(len(w_func1[t1])):
                    result[waiting_time, i] += pmf1[t1, i] * pmf2[t2, i] * output[i]
                
    return result






