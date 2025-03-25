import time
import warnings
from copy import deepcopy
from collections.abc import Iterable
import logging

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
try:
    import cupy as cp
    _cupy_exist = True
except (ImportError, ModuleNotFoundError):
    _cupy_exist = False

from protocol_units import join_links_compatible
from protocol_units_efficient import join_links_efficient
from utility_functions import secret_key_rate, ceil, bell_to_fid, find_heading_zeros_num, matrix_to_werner, werner_to_matrix, get_fidelity
from logging_utilities import log_init, create_iter_kwargs, save_data
from repeater_mc import repeater_mc, plot_mc_simulation


__all__ = ["RepeaterChainSimulation", "compute_unit", "plot_algorithm",
           "join_links_compatible", "repeater_sim"]


class RepeaterChainSimulation():
    def __init__(self):
        self.use_fft = True
        self.use_gpu = False
        self.gpu_threshold = 1000000
        self.efficient = False
        self.zero_padding_size = None
        self._qutip = False

    def iterative_convolution(self,
            func, shift=0, first_func=None, p_swap=None):
        """
        Calculate the convolution iteratively:
        first_func * func * func * ... * func
        It returns the sum of all iterative convolution:
        first_func + first_func * func + first_func * func * func ...

        Parameters
        ----------
        func: array-like
            The function to be convolved in array form.
            It is always a probability distribution.
        shift: int, optional
            For each k the function will be shifted to the right. Using for
            time-out mt_cut.
        first_func: array-like, optional
            The first_function in the convolution. If not given, use func.
            It can be different because the first_func is
            `P_s` and the `func` P_f.
            It is upper bounded by 1.
            It can be a probability, or an array of states.
        p_swap: float, optimal
            Entanglement swap success probability.

        Returns
        -------
        sum_convolved: array-like
            The result of the sum of all convolutions.
        """
        # is_dm is True if the first_func is a density matrix
        if first_func is None or len(first_func.shape) == 2:  
            is_dm = False
        else:
            is_dm = True
        trunc = len(func)
        # determine the required number of convolution
        if shift != 0:
            # cut-off is added here.
            # because it is a constant, we only need size/mt_cut convolution.
            max_k = int(np.ceil((trunc/shift)))
        else:
            max_k = trunc
        sum_func = 0
        for sublist in func:
            sum_func += sublist[0]
        #if p_swp is a list, extract the first element
        if p_swap is not None:
            if isinstance(p_swap, list):
                p_swap = p_swap[0]
            pf = sum_func * (1 - p_swap)
        else:
            pf = sum_func
        with np.errstate(divide='ignore'):
            if pf <= 0.:  # pf ~ 0 and round-off error
                max_k = trunc
            else:
                max_k = min(max_k, (-52 - np.log(trunc))/ np.log(pf))
        if max_k > trunc:
            print(max_k)
            print(trunc)
        max_k = max_k.real
        max_k = int(max_k)

        # Transpose the array of state to the shape (1, 4, trunc)
        if first_func is None:
            first_func = func
        if not is_dm:
            first_func = first_func.reshape((trunc, 16, 1))
        first_func = np.transpose(first_func, (1, 2, 0))
        func = func.reshape((trunc,16, 1))
        func = np.transpose(func, (1, 2, 0))
        # Convolution
        result = np.empty(first_func.shape, first_func.dtype)
        for i in range(first_func.shape[0]):
            for j in range(first_func.shape[1]):
                result[i][j] = self.iterative_convolution_helper(
                    func[i][j], first_func[i][j], trunc, shift, p_swap, max_k)
        # Permute the indices back
        result = np.transpose(result, (2, 0, 1))
        if not is_dm:
            result = result.reshape([trunc, 16])
        return result

    def iterative_convolution_helper(
            self, func, first_func, trunc, shift, p_swap, max_k):
        # initialize the result array
        sum_convolved = np.zeros(trunc, dtype=first_func.dtype)
        if p_swap is not None:
            sum_convolved[:len(first_func)] = p_swap * first_func
        else:
            sum_convolved[:len(first_func)] = first_func

        if shift <= trunc:
            zero_state = np.zeros(shift, dtype=func.dtype)
            func = np.concatenate([zero_state, func])[:trunc]
        

        # decide what convolution to use and prepare the data
        convolved = first_func
        if self.use_fft: # Use geometric sum in Fourier space
            shape = 2 * trunc - 1
            # The following is from SciPy, they choose the size to be 2^n,
            # It increases the accuracy.
            if self.zero_padding_size is not None:
                shape = self.zero_padding_size
            else:
                shape = 2 ** np.ceil(np.log2(shape)).astype(int)
            if self.use_gpu and not _cupy_exist:
                logging.warning("CuPy not found, using CPU.")
                self.use_gpu = False
            if self.use_gpu and shape > self.gpu_threshold:
                # transfer the data to GPU
                sum_convolved = cp.asarray(sum_convolved)
                convolved = cp.asarray(convolved)
                func = cp.asarray(func)
            if self.use_gpu and shape > self.gpu_threshold:
                # use CuPy fft
                ifft = cp.fft.ifft
                fft = cp.fft.fft
                to_real = cp.real
            else:
                # use NumPy fft
                ifft = np.fft.ifft
                fft = np.fft.fft
                to_real = np.real
            convolved_fourier = fft(convolved, shape)
            func_fourier = fft(func, shape)
            if p_swap is not None:
                result= ifft(
                    p_swap*convolved_fourier / (1 - (1-p_swap) * func_fourier))
            else:
                result= ifft(convolved_fourier / (1 - func_fourier))
            # validity check
            last_term = abs(result[-1])
            # if last_term > 10e-16:
            #     logging.warning(
            #         f"The size of zero-padded array, shape={shape}, "
            #         "for the Fourier transform is not big enough. "
            #         "The resulting circular convolution might contaminate "
            #         "the distribution."
            #         f"The deviation is as least {float(last_term):.0e}.")
            result = to_real(result[:trunc])
            if self.use_gpu and shape > self.gpu_threshold:
                result = cp.asnumpy(result)

        else:  # Use exact convolution
            zero_state = np.zeros(trunc - len(convolved), dtype=convolved.dtype)
            convolved = np.concatenate([convolved, zero_state])
            
            for k in range(1, max_k):
                convolved = np.convolve(convolved[:trunc], func[:trunc])
                if p_swap is not None:
                    coeff = p_swap*(1-p_swap)**(k)
                    sum_convolved += coeff * convolved[:trunc]
                else:
                    sum_convolved += convolved[:trunc]
            result = sum_convolved
        return result

    def entanglement_swap(self,
        pmf1, func1, pmf2, func2,
        cutoff, cut_type, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
        """
        Calculate the waiting time and average parameters with time-out
        for entanglement swap.

        Parameters
        ----------
        pmf1, pmf2: array-like 2-D
            The waiting time distribution of the two input links.
            Even though the list is 2-D, each sublist shares the same values as 4 elements
        w_func1, w_func2: array-like 2-D
            The Bell Diagonal coefficients as function of T of the two input links.
        cutoff: int or float
            The memory time cut-off,  parameter cut-off, or
            run time cut-off.
        cut_type: str
            `memory_time`, `fidelity` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the entanglement swap.
        w_func: array-like 2-D
            The Bell diagonal coefficeients as function of T of the entanglement swap.
        """
        if self.efficient and cut_type == "memory_time":
            join_links = join_links_efficient
            if self._qutip:
                # only used for testing, very slow
                join_links_state = join_links_matrix_qutip
            else:
                join_links_state = join_links_efficient
        else:
            join_links = join_links_compatible
            join_links_state = join_links_compatible
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0

        # P'_f
        pf_cutoff = join_links(
            pmf1, pmf2, func1, func2, ycut=False,
            cutoff=cutoff, cut_type=cut_type, 
            evaluate_func="1", depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # P'_s
        ps_cutoff = join_links(
            pmf1, pmf2, func1, func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type, 
            evaluate_func="1", depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # P_f or P_s 
        pmf_cutoff = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=ps_cutoff)
        del ps_cutoff
        # Pr(Tout = t)
        pmf_swap = self.iterative_convolution(
            pmf_cutoff, shift=0, p_swap=1/4)
        
        # Wsuc * P_s
        state_suc = join_links_state(
            pmf1, pmf2, lambda_func1=func1, lambda_func2=func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="w1w2", depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # Wprep * Pr(Tout = t)
        state_prep = self.iterative_convolution(
            pf_cutoff,
            shift=shift, first_func=state_suc)
        del pf_cutoff, state_suc
        # Wout * Pr(Tout = t)
        state_out = self.iterative_convolution(
            pmf_cutoff, shift=0,
            first_func=state_prep, p_swap=1/4)
        del pmf_cutoff

        

        # Checks whther state_out is 1D or 3D 
        with np.errstate(divide='ignore', invalid='ignore'):
            if len(state_out.shape) == 2 and state_out.shape[1] == 16:
                for i in range(1, len(state_out)):
                    state_out[i] = state_out[i] / pmf_swap[i]
                    for j in range(0, state_out.shape[1]):
                        state_out[i][j] = np.where(np.isnan(state_out[i][j]), 1., state_out[i][j] ) #if nan, replace state_out with 1
            else:
                raise ValueError("The state_out is not in the correct shape.")
                state_out = np.transpose(state_out, (1, 2, 0))
                state_out[:,:,1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
                state_out = np.transpose(state_out, (2, 1, 0))
        return pmf_swap, state_out


    def distillation(self,
            pmf1, func1, pmf2, func2,
            cutoff, cut_type, depolar_rate=0., dephase_rate=0., amplitude_damping_rate=0., bit_phase_flip_rate=0.):
        """
        Calculate the waiting time and average Bell diagonal coefficients
        with time-out for the distillation.

        Parameters
        ----------
        pmf1, pmf2: array-like 2-D
            The waiting time distribution of the two input links.
        w_func1, w_func2: array-like 2-D
            The Bell diagonal coefficients as function of T of the two input links.
        cutoff: int or float
            The memory time cut-off, fidelity cut-off, or 
            run time cut-off.
        cut_type: str
            `memory_time`, `fidelity` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the distillation.
        w_func: array-like 2-D
            The Bell diagonal coeffiicient as function of T of the distillation.
        """
        if self.efficient and cut_type == "memory_time":
            join_links = join_links_efficient
        else:
            join_links = join_links_compatible
            join_links_state = join_links_compatible
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0

        # P'_f  cutoff attempt when cutoff fails
        pf_cutoff = join_links(
            pmf1, pmf2, func1, func2, ycut=False,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="1", depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # P'_ss  cutoff attempt when cutoff and dist succeed
        pss_cutoff = join_links(
            pmf1, pmf2, func1, func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="0.5+0.5w1w2",  depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        pss_cutoff_link = join_links(
            pmf1, pmf2, func1, func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="1",  depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # P_s  dist attempt when dist succeeds
        ps_dist = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=pss_cutoff)
        ps_dist_link = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=pss_cutoff_link)
        del pss_cutoff
        # P'_sf  cutoff attempt when cutoff succeeds but dist fails
        psf_cutoff = join_links(
            pmf1, pmf2, func1, func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="0.5-0.5w1w2", depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate) 
        # P_f  dist attempt when dist fails
        pf_dist = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=psf_cutoff)
        del psf_cutoff
        # Pr(Tout = t)
        pmf_dist = self.iterative_convolution(
            pf_dist, shift=0,
            first_func=ps_dist)
        pmf_dist_link = self.iterative_convolution(
            pf_dist, shift=0,
            first_func=ps_dist_link)
        del ps_dist

        # Wsuc * P'_ss
        state_suc = join_links_state(
            pmf1, pmf2, func1, func2, ycut=True,
            cutoff=cutoff, cut_type=cut_type,
            evaluate_func="w1+w2+4w1w2",  depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # Wprep * P_s
        state_prep = self.iterative_convolution(
            pf_cutoff, shift=shift,
            first_func=state_suc)
        del pf_cutoff, state_suc
        # Wout * Pr(Tout = t)
        state_out = self.iterative_convolution(
            pf_dist, shift=0,
            first_func=state_prep)
        del pf_dist, state_prep


        with np.errstate(divide='ignore', invalid='ignore'):
            if len(state_out.shape) == 2 and state_out.shape[1] == 16:
                for i in range(1, len(state_out)):    
                    for j in range(0, state_out.shape[1]):
                        state_out[i][j] = state_out[i][j] / pmf_dist_link[i][j]
                        state_out[i][j] = np.where(np.isnan(state_out[i][j]), 1., state_out[i][j] ) #if nan, replace state_out with 1
            else:
                raise ValueError("The state_out is not in the correct shape.")
                state_out = np.transpose(state_out, (1, 2, 0))
                state_out[:,:,1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
                state_out = np.transpose(state_out, (2, 1, 0))
        return pmf_dist, state_out


    def compute_unit(self,
            parameters, pmf1, func1, pmf2=None, func2=None,
            unit_kind="swap", step_size=1):
        """
        Calculate the the waiting time distribution and
        the parameter of a protocol unit swap or distillation.
        Cut-off is built in swap or distillation.

        Parameters
        ----------
        parameters: dict
            A dictionary contains the parameters of
            the repeater and the simulation.
        pmf1, pmf2: array-like 2-D
            The waiting time distribution of the two input links.
        lambda_func1, lambda_func2: array-like 2-D
            The Bell parameter as function of T of the two input links.
        unit_kind: str
            "swap" or "dist"

        Returns
        -------
        t_pmf: array-like 1-D
            The output waiting time 
        w_func: array-like 2-D
            The output parameter
        """
        if pmf2 is None:
            pmf2 = pmf1
        if func2 is None:
            func2 = func1
        p_gen = parameters["p_gen"]
        coeff = parameters["coefficients"]
        cut_type = parameters.get("cut_type", "memory_time")
        depolar_rate = parameters.get("depolarizing_rate", 0.)
        dephase_rate = parameters.get("dephasing_rate", 0.)
        amplitude_damping_rate = parameters.get("amplitude_damping_rate", 0.)
        bit_phase_flip_rate = parameters.get("bit_phase_flip_rate", 0.)
        if "cutoff" in parameters.keys():
            cutoff = parameters["cutoff"]
        elif cut_type == "memory_time":
            cutoff = parameters.get("mt_cut", np.iinfo(int).max)
        elif cut_type == "fidelity":
            cutoff = parameters.get("w_cut", 1.0e-16)  # shouldn't be zero
            if cutoff == 0.:
                cutoff = 1.0e-16
        elif cut_type == "run_time":
            cutoff = parameters.get("rt_cut", np.iinfo(int).max)
        else:
            cutoff = np.iinfo(int).max

        # type check
        if not np.isreal(p_gen):
            raise TypeError("p_gen must be a float number.")
        if cut_type in ("memory_time", "run_time") and not np.issubdtype(type(cutoff), int):
            raise TypeError(f"Time cut-off must be an integer. not {cutoff}")
        if cut_type == "fidelity" and not (cutoff >= 0. or cutoff < 1.):
            raise TypeError(f"Fidelity cut-off must be a real number between 0 and 1.")
        #if sum of lamdas is not 1.0 raise error
        # if not np.isclose(np.sum(coeff), 1.0):
        #     raise TypeError(f"Invalid lambda parameters, sum of lambdas must be 1.0")
        # swap or distillation for next level
        if unit_kind == "swap":
            pmf, func = self.entanglement_swap(
                pmf1, func1, pmf2, func2,
                cutoff=cutoff, cut_type=cut_type, depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        elif unit_kind == "dist":
            pmf, func = self.distillation(
                pmf1, func1, pmf2, func2,
                cutoff=cutoff, cut_type=cut_type, depolar_rate=depolar_rate, dephase_rate=dephase_rate, amplitude_damping_rate=amplitude_damping_rate, bit_phase_flip_rate=bit_phase_flip_rate)
        # erase ridiculous parameters,
        # it can happen when the probability is too small ~1.0e-20.
        func = np.where(np.isnan(func), 1., func)
        func = np.clip(func, 0., 1.0)

        # check probability coverage
        coverage = np.sum(sublist[0] for sublist in pmf)
        if coverage < 0.245:
            logging.warning(
                "The truncation time only covers {:.2f}% of the distribution, "
                "please increase t_trunc.\n".format(
                    coverage*100))
        
        return pmf, func

    def general_protocol(self, parameters, all_level=False):

        parameters = deepcopy(parameters)
        protocol = parameters["protocol"]
        p_gen = parameters["p_gen"]
        coeff = parameters["coefficients"]
        depolar_rate = parameters.get("depolarizing_rate", 0.)
        dephase_rate = parameters.get("dephasing_rate", 0.)
        amplitude_damping_rate = parameters.get("amplitude_damping_rate", 0.)
        bit_phase_flip_rate = parameters.get("bit_flip_rate", 0.)
        if "tau" in parameters:
            parameters["mt_cut"] = parameters.pop("tau")
        if "cutoff_dict" in parameters.keys():
            cutoff_dict = parameters["cutoff_dict"]
            mt_cut = cutoff_dict.get("memory_time", np.iinfo(int).max)
            w_cut = cutoff_dict.get("fidelity", 1.e-8)
            rt_cut = cutoff_dict.get("run_time", np.iinfo(int).max)
        else:
            mt_cut = parameters.get("mt_cut", np.iinfo(int).max)
            w_cut = parameters.get("w_cut", 1.e-8)
            rt_cut = parameters.get("rt_cut", np.iinfo(int).max)
        if "cutoff" in parameters:
            cutoff = parameters["cutoff"]
        if not isinstance(mt_cut, Iterable):
            mt_cut = (mt_cut,) * len(protocol)
        else:
            mt_cut = tuple(mt_cut)
        if not isinstance(w_cut, Iterable):
            w_cut = (w_cut,) * len(protocol)
        else:
            w_cut = tuple(w_cut)
        if not isinstance(rt_cut, Iterable):
            rt_cut = (rt_cut,) * len(protocol)
        else:
            rt_cut = tuple(rt_cut)
        
        t_trunc = parameters["t_trunc"]

        # elementary link
        t_list = np.arange(1, t_trunc) # t_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pmf = p_gen * (1 - p_gen)**(t_list - 1) # pmf = [0.1, 0.09, 0.081, 0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.043046721, 0.0387420489]
        pmf = np.concatenate((np.array([0.]), pmf)) # pmf = [0.0, 0.1, 0.09, 0.081, 0.0729, 0.06561, 0.059049, 0.0531441, 0.04782969, 0.043046721, 0.0387420489]
        pmf = np.tile(pmf[:, np.newaxis], len(coeff)) # pmf = [[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1], [0.09, 0.09, 0.09, 0.09], [0.081, 0.081, 0.081, 0.081], [0.0729, 0.0729, 0.0729, 0.0729], [0.06561, 0.06561, 0.06561, 0.06561], [0.059049, 0.059049, 0.059049, 0.059049], [0.0531441, 0.0531441, 0.0531441, 0.0531441], [0.04782969, 0.04782969, 0.04782969, 0.04782969], [0.043046721, 0.043046721, 0.043046721, 0.043046721], [0.0387420489, 0.0387420489, 0.0387420489, 0.0387420489]]
        func = np.array([coeff] * t_trunc)
        if all_level:
            full_result = [(pmf, func)]
        func = func.astype(np.complex128)

        total_step_size = 1
        # protocol unit level by level
        for i, operation in enumerate(protocol):
            if "cutoff" in parameters and isinstance(cutoff, Iterable):
                parameters["cutoff"] = cutoff[i]
            parameters["mt_cut"] = mt_cut[i]
            parameters["w_cut"] = w_cut[i]
            parameters["rt_cut"] = rt_cut[i]
        
            if operation == 0:
                pmf, func = self.compute_unit(
                    parameters, pmf, func, unit_kind="swap", step_size=total_step_size)
            elif operation == 1:
                pmf, func = self.compute_unit(
                    parameters, pmf, func, unit_kind="dist", step_size=total_step_size)
            if all_level:
                full_result.append((pmf, func))
        final_pmf = [sublist[0] for sublist in pmf]
        final_pmf = np.array(final_pmf, dtype=complex)
        final_pmf = final_pmf[np.isreal(final_pmf)].real
        final_lambda_func = func

        if all_level:
            return full_result
        else:
            return final_pmf, final_lambda_func



def compute_unit(
        parameters, pmf1, w_func1, pmf2=None, w_func2=None,
        unit_kind="swap", step_size=1):
    """
    Functional warpper for compute_unit
    """
    simulator = RepeaterChainSimulation()
    return simulator.compute_unit(
        parameters=parameters, pmf1=pmf1, w_func1=w_func1, pmf2=pmf2, w_func2=w_func2, unit_kind=unit_kind, step_size=step_size)


def repeater_sim(parameters, all_level=False):
    """
    Functional warpper for nested_protocol
    """
    simulator = RepeaterChainSimulation()

    if isinstance(parameters["lambdas"], Iterable):  
        return simulator.general_protocol(parameters=parameters, all_level=all_level)


def plot_algorithm(pmf, fid_func, axs=None, t_trunc=None, legend=None):
    """
    Plot the waiting time distribution and Werner parameters
    """
    cdf = np.cumsum(pmf)
    if t_trunc is None:
        try:
            t_trunc = np.min(np.where(cdf >= 0.997))
        except ValueError:
            t_trunc = len(pmf)
    pmf = pmf[:t_trunc]
    fid_func = fid_func[:t_trunc]
    fid_func[0] = np.nan

    axs[0][0].plot((np.arange(t_trunc)), np.cumsum(pmf))

    axs[0][1].plot((np.arange(t_trunc)), pmf)
    axs[0][1].set_xlabel("Waiting time $T$")
    axs[0][1].set_ylabel("Probability")

    axs[1][0].plot((np.arange(t_trunc)), fid_func)
    axs[1][0].set_xlabel("Waiting time $T$")
    axs[1][0].set_ylabel("Werner parameter")

    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
