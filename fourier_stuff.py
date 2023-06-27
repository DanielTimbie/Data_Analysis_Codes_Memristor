# -*- coding: utf-8 -*-
"""
This class contains Cobi's attempts to improve the fourier-analysis code.
Specifically, to make sure that the code works even for data with dropped
frames in it.

All these transforms are formatted to take in inputs as REAL CURRENT and output
Fourier COMPONENTS.  So output[0] is the average of the input and output[k] is
the amplitude in front of sin(2*pi*t*freq[k]) for frequencies in Hz.

If you want units like "root-mean-square units per square root of Hz", you need
to convert using :func:`ComponentsToSpectralDensity`.

"""

# from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt







def StandardFourier(ys,fps):
    """Finds the fourier components of the given time-series
    
    Args:
        ys (array): Time-series of current in units of Amps
        fps (float): Frame rate of the measurement, in Hz.
    Returns:
        array: One-sided fourier transform (positive frequencies only, plus DC)
    """
    fs = np.linspace(0,fps/2,len(ys)//2,endpoint=False)  #frequencies, in Hz, of half-spectrum
    fft = np.abs(np.fft.fft(ys)) / len(ys)
    fft = fft[:len(ys)//2] * np.sqrt(2)  #take only half the spectrum, so double the "power"-sqrt(2) times the amplitude
    fft[0] /= np.sqrt(2)  #undo the multiplication by 2 for the DC offset
    return fs,fft

def HanningApply(ys,show=False):
    """Applies a normalized Hanning filter to the given data, so that a fourier
    transform can be taken of it later.

    Args:
        ys (array): Time-series of current in units of Amps.
        show (bool): Whether to plot the raw and filtered data.
    Returns:
        array: filtered and normalized time-series of current, in units of Amps
    """
    smoothed = ys*np.hanning(len(ys))
    if show:
        plt.figure()
        plt.plot(ys)
        plt.plot(smoothed)
    return smoothed*1.63 #Energy normalization factor

def InterpFourierComponents(targetfs,knownfs,knowncomponents):
    """Interpolate fourier components to find components at unknown frequencies.

    I tested this function, it preserves the sum of all frequency components.
    
    Args:
        targetfs (array): The frequencies (in Hz) where we want fourier
            components.
        knownfs (array): The frequencies (in Hz) where we already have 
            fourier components.
        knowncomponents (array): The fourier components we already know.
            Should correspond to `knownfs`.
    Returns:
        array: Fourier components corresponding to `targetfs`.
    """
    fours_i = np.interp(targetfs[1:],knownfs[1:],knowncomponents[1:])  #interpolate
    fours_i *= (targetfs[1]-targetfs[0])/(knownfs[1]-knownfs[0])       #rescale to freq bin size
    fours_i = np.concatenate([[knowncomponents[0]],fours_i])                    #DC not really part of spectrum
    return fours_i

def ComponentsToSpectralDensity(freqs,components):
    """Converts amplitude of sine-wave fourier components into
    RMS-per-sqrt(Hz) units.
    
    Args:
        freqs (array): The frequencies (in Hz) of the fourier components. 
            Assumes these are already only one-sided and absolute valued.
        components (array): The corresponding fourier components. 
    Returns:
        array: The spectral density, in units of RMS-per-sqrt(Hz). (No, not
        Amps_rms, just RMS.)
    """
    fps = (freqs[1]-freqs[0])
    density = components*np.sqrt(1/fps)  
    return density

###---working toward non-uniform fourier transform
"""from: https://notebook.community/jakevdp/nufftpy/NUFFT-Numba"""

def nufftfreqs(M, df=1):
    """Compute the frequency range used in :func:`nufft` for `M` frequency
    bins with a spacing of `df` Hz."""
    return df * np.arange(-(M // 2), M - (M // 2))


def nudft(x, y, M, df=1.0, iflag=1):
    """Non-Uniform Direct Fourier Transform.
    See https://notebook.community/jakevdp/nufftpy/NUFFT-Numba for details.
    """
    sign = -1 if iflag < 0 else 1
    fft = np.dot(y, np.exp(sign * 1j * 2 * np.pi * nufftfreqs(M, df) * x[:, np.newaxis]))
    fft[M//2] /= 2*np.pi
    return fft


def SlowNonUniform(times,frames,fps,iflag=1):
    """Non-uniform fourier transform, naive (slow) implementation. Verified but slow.
    
    Args:
        times (array): The times at which the time-series occur, in seconds.
        frames (array): The output values at each time.
        fps (float): Frame rate of the measurement, in Hz.
        iflag (int): Controls the sign.
    
    Returns:
        array,array: frequencies in Hz, fourier components
    """
    # freqnum=len(ys)/2,df=fps/len(ys)  # freqs = df * np.arange(-freqnum, freqnum)
    sign = -1 if iflag < 0 else 1
    freqs = np.linspace(0,fps/2,len(frames)//2,endpoint=False)
    fft = np.dot(frames, np.exp(sign * 2j * np.pi * freqs * times[:, np.newaxis])) / len(times)
    #we only calculated this for the positive freqs, so we need to double the power
    fft[1:] *= 2  #DC offset at [0] doesn't need doubling
    return freqs,abs(fft)
    

# =============================================================================
#
# ###---Attempts to vectorize the nun-uniform discrete fourier transform.
# ###---This section is commented out because those attemps mostly failed.
# ###---But I leave them here as a starting point if anyone else wants to work
# ###---on this later
#
# def _compute_grid_params(M, eps):
#     """NOT YET FULLY FUNCTIONAL"""
#     # Choose Msp & tau from eps following Dutt & Rokhlin (1993)
#     if eps <= 1E-33 or eps >= 1E-1:
#         raise ValueError("eps = {0:.0e}; must satisfy "
#                          "1e-33 < eps < 1e-1.".format(eps))
#     ratio = 2 if eps > 1E-11 else 3
#     Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
#     Mr = max(ratio * M, 2 * Msp)
#     lambda_ = Msp / (ratio * (ratio - 0.5))
#     # tau = np.pi * lambda_ / M ** 2
#     tau = 0.5 * lambda_ / M ** 2
#     return Msp, Mr, tau
# 
# 
# def nufft_numpy(x, y, numfreqbins, df=1.0, iflag=1, eps=1E-15):
#     """FAST Non-Uniform Fourier Transform, vectorized with numpy. NOT YET FULLY FUNCTIONAL"""
#     Msp, Mr, tau = _compute_grid_params(numfreqbins, eps)
#     print(Msp,Mr,tau)
#     N = len(x)
# 
#     # Construct the convolved grid ftau:
#     # this replaces the loop used above
#     ftau = np.zeros(Mr, dtype=y.dtype)
#     # hx = 2 * np.pi / Mr
#     # xmod = (x * df) % (2 * np.pi)
#     hx = 1/Mr
#     xmod = (x * df) % 1
#     m = 1 + (xmod // hx).astype(int)
#     mm = np.arange(-Msp, Msp)
#     mpmm = m + mm[:, np.newaxis]  #array of shape (2*Msp,len(x))
#     spread = y * np.exp(-0.25 * 2*np.pi* (xmod - hx * mpmm) ** 2 / tau)
#     np.add.at(ftau, mpmm % Mr, spread)  #np.add.at(x,i,y) is similar to x[i] += y
# 
#     # Compute the FFT on the convolved grid
#     if iflag < 0:
#         Ftau = (1 / Mr) * np.fft.fft(ftau)
#     else:
#         Ftau = np.fft.ifft(ftau)
#     Ftau = np.concatenate([Ftau[-(numfreqbins//2):], Ftau[:numfreqbins//2 + numfreqbins % 2]])  #reorder b/c np.fft return is positive then negative
# 
#     # Deconvolve the grid using convolution theorem
#     k = nufftfreqs(numfreqbins,df=df)
#     # return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau
#     return (1 / N) * np.sqrt(0.5 / tau) * np.exp(tau * k ** 2) * Ftau
# 
# 
# 
# ###---safe copy of this function while I go and break the other version
# def nufft_numpy_copy(x, y, numfreqbins, df=1.0, iflag=1, eps=1E-15):
#     """FAST Non-Uniform Fourier Transform, vectorized with numpy"""
#     Msp, Mr, tau = _compute_grid_params(numfreqbins, eps)
#     print(Msp,Mr,tau)
#     N = len(x)
# 
#     # Construct the convolved grid ftau:
#     # this replaces the loop used above
#     ftau = np.zeros(Mr, dtype=y.dtype)
#     hx = 2 * np.pi / Mr
#     xmod = (x * df) % (2 * np.pi)
#     m = 1 + (xmod // hx).astype(int)
#     mm = np.arange(-Msp, Msp)
#     mpmm = m + mm[:, np.newaxis]
#     spread = y * np.exp(-0.25 * (xmod - hx * mpmm) ** 2 / tau)
#     np.add.at(ftau, mpmm % Mr, spread)
# 
#     # Compute the FFT on the convolved grid
#     if iflag < 0:
#         Ftau = (1 / Mr) * np.fft.fft(ftau)
#     else:
#         Ftau = np.fft.ifft(ftau)
#     Ftau = np.concatenate([Ftau[-(numfreqbins//2):], Ftau[:numfreqbins//2 + numfreqbins % 2]])
# 
#     # Deconvolve the grid using convolution theorem
#     k = nufftfreqs(numfreqbins)
#     return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau
# =============================================================================





def FourierFromSnippets(ys,fps,bounds,show=False,usehann=False,weighavrg=True):
    """Take the fourier transform of each good segment of the data, then
    average the spectra together. Interpolates so the frequency bins match.
    
    Args:
        ys (array): The data time-series. Presumably pixel output of some sort.
        fps (float): The frame rate at which the data was taken, in Hz.
        bounds (array): a list of indices giving the borders of the good 
            segments of the data.  The first segment of data between
            `bounds` [0] and `bounds` [1] is good, the next segment is
            bad, etc.
        show (bool): Whether to show a plot of the process.
        usehann (bool): Whether to apply the Hanning filter or not.
        weighavrg (bool): Whether to normalize each segment of spectrum by the
            length of the segment (True) or average them without normalizing
            first (False). Cobi believes it is always correct to normalize.
            
    Returns:
        array,array: frequencies in Hz, fourier components
    """
    assert(bounds[0]>=0)
    assert(bounds[-1]<=len(ys))
    assert(len(bounds)%2==0)
    #split according to bounds. region before first bound is bad, so take every
    #other section starting with second section
    goodsegs = np.split(ys,bounds)[1::2]
    #find length of longest segment. that gives the bin spacing we'll interpolate everything into
    maxnum = np.max([len(seg) for seg in goodsegs])
    freqs_base = np.linspace(0,fps/2,maxnum//2,endpoint=False)  #frequencies, in Hz, of half-spectrum
    #find sum of all length segments, for normalization later
    totallength = np.sum([len(s) for s in goodsegs])
    #take fourier of each segment independently
    fouriers = []
    
    while len(goodsegs)>0:
        seg = goodsegs.pop()
        #if segment is too short, just skip it. nothing worthwile gonna happen here
        if len(seg)<10:
            continue
        #if segment is too long, fragment it first
        if len(seg)>20000:
            goodsegs.append(seg[:20000])
            goodsegs.append(seg[20000:])
            continue #put fragments back in heap and then grab next item from queue
        if usehann:
            freqs,fours = StandardFourier( HanningApply(seg) ,fps)
        else:
            freqs,fours = StandardFourier( seg ,fps)    
        interp = InterpFourierComponents(freqs_base,freqs,fours)  #interpolate into base frequency
        if weighavrg:
            fouriers.append(interp * len(seg) / totallength)            ##---------weigh each segment by its length
        else:
            fouriers.append(interp)

    #plot these independent fourier transforms, if requested
    if show:
        fig,ax = plt.subplots()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("components")
        ax.set_yscale("log")
        freqs,fours = StandardFourier(ys,fps)
        ax.plot(freqs,fours,"-r",label="original")
        for k in range(len(fouriers)):
            f = fouriers[k] * totallength / len(goodsegs[k])  #undo weighing for display purposes
            ax.plot(freqs_base,f, ".-",label="%i-%i" %(bounds[k],bounds[k+1]))
        ax.set_title("Fourier components of each segment of the data")
        ax.legend()
    
    #average the two partial-sections back together
    if weighavrg:
        rejoined = np.sum(fouriers,axis=0)  #sum, not average, because each weighted interp is divided by length already
    else:
        rejoined = np.average(fouriers,axis=0)  #arithmetic average
    return freqs_base,rejoined


def FourierAvoidingNanSegments(ys,fps,show=False,usehann=False,weighavrg=True,ax=None,spectraldensity=True):
    """Returns the Fourier components of the given time-series with dropped 
    frames (NaN values). 
    
    This function uses :func:`FourierFromSnippets` to take a separate fourier
    transforms of each continuous segment and average the resulting spectra.
    
    Args:
        ys (array): The data time-series, in units of Amps.
        fps (float): The frame rate at which the data was taken, in Hz.
        show (bool): Whether to show a plot of the process.
        usehann (bool): Whether to apply the Hanning filter or not.
        weighavrg (bool): Whether to normalize each segment of spectrum by the
            length of the segment (True) or average them without normalizing
            first (False). Cobi believes it is always correct to normalize.
        ax (pyplot axis): The pyplot axis object to plot on.  If none,
            opens a fresh figure instead.
        spectraldensity (bool): Whether to return in units of Amps_rms/sqrt(Hz)
            (True) or units of Amps (False).
            
    Returns:
        array,array: frequencies in Hz, fourier transform
    """
    #find the borders where nans are
    nans = np.isnan(ys)
    borders = np.nonzero(nans[1:] != nans[:-1])[0]  #list of last index in each good-or-bad section
    borders += 1  #list of index of FIRST frame in each section
    #add 0 and end if needed. Section b/w 1st and 2nd index should be good, same with b/w 3rd and 4th, etc
    if not nans[0]:
        borders = np.insert(borders,0,0)  #add 0 to beginning if data[0] is valid data
    if not nans[-1]:
        borders = np.append(borders,len(nans))  #add -1 to end if data[-1] is valid data
    freqs_base,rejoined = FourierFromSnippets(ys,fps=fps,bounds=borders,show=False,usehann=usehann,weighavrg=weighavrg)
    if show:
        if ax is None:
            fig_f,ax_f = plt.subplots()
        else:
            ax_f = ax
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("components" if not spectraldensity else "current_rms/sqrt(Hz)")
        ax_f.set_yscale("log")
        clean = [i for i in ys if (not np.isnan(i))]
        if usehann:
            clean = HanningApply(clean)
        freqs,fours = StandardFourier( clean,fps=fps)
        if spectraldensity:
            fours = ComponentsToSpectralDensity(freqs,fours)
        ax_f.plot(freqs[2:],fours[2:],"-r",label="skipping nans without realizing")
        lbl = "("
        k=0
        while k < len(borders):
            lbl+="%i-%i," %(borders[k],borders[k+1])
            k+=2
        lbl = lbl[:-1] +")"
        toshow = rejoined
        if spectraldensity:
            toshow = ComponentsToSpectralDensity(freqs_base,rejoined)
        ax_f.plot(freqs_base[2:],toshow[2:],"-k",label="segments "+lbl)
        ax_f.legend(title="Hanning filter: "+str(usehann)+"   |   weighed avrg: "+str(weighavrg))
    #return the results
    if spectraldensity:
        density = ComponentsToSpectralDensity(freqs_base,rejoined)
        return freqs_base,density
    else:
        return freqs_base,rejoined
        
   



if __name__ == '__main__':

    #giant pile of real data (5CML19\Noise180K_20210727_frameskiptest\H191V54_20210727_180K_4590FPS_Gain0_600V_0fW_202uS pix 200,60 [1750:5000]
    data = ['nan','nan','nan','nan','nan','nan','nan','nan','nan',3326 ,3322 ,3334 ,3337 ,3329 ,3329 ,
            3309 ,3314 ,3346 ,3333 ,3330 ,3326 ,3318 ,3336 ,3334 ,3322 ,3327 ,3322 ,3325 ,3337 ,3332 ,
            3329 ,3319 ,3318 ,3305 ,3311 ,3315 ,3301 ,3307 ,3300 ,3317 ,3304 ,3296 ,3297 ,3303 ,3289 ,
            3301 ,3298 ,3322 ,3319 ,3316 ,3313 ,3337 ,3324 ,3329 ,3323 ,3318 ,3331 ,3338 ,3341 ,3334 ,
            3313 ,3325 ,3332 ,3334 ,3317 ,3332 ,3337 ,3329 ,3314 ,3318 ,3322 ,3315 ,'nan','nan','nan',
            'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan',
            'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan',
            'nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan',
            'nan','nan','nan','nan','nan','nan','nan','nan','nan',3304 ,3314 ,3333 ,3332 ,3346 ,3336 ,
            3335 ,3339 ,3325 ,3329 ,3317 ,3331 ,3319 ,3336 ,3335 ,3317 ,3316 ,3319 ,3317 ,3321 ,3314 ,
            3308 ,3299 ,3283 ,3306 ,3306 ,3300 ,3313 ,3328 ,3319 ,3311 ,3319 ,3336 ,3329 ,3328 ,3330 ,
            3331 ,3330 ,3329 ,3324 ,3323 ,3325 ,3328 ,3334 ,3330 ,3320 ,3303 ,3326 ,3315 ,3314 ,3307 ,
            3311 ,3322 ,3318 ,3313 ,3316 ,3320 ,3309 ,3310 ,3302 ,3307 ,3303 ,3299 ,3298 ,3291 ,3289 ,
            3302 ,3302 ,3296 ,3304 ,3322 ,3325 ,3312 ,3316 ,3326 ,3321 ,3306 ,3324 ,3313 ,3318 ,3316 ,
            3334 ,3342 ,3340 ,3326 ,3337 ,3334 ,3346 ,3338 ,3341 ,3335 ,3348 ,3331 ,3326 ,3348 ,3319 ,
            3331 ,3333 ,3324 ,3318 ,3298 ,3311 ,3313 ,3292 ,3296 ,3304 ,3299 ,3310 ,3343 ,3315 ,3317 ,
            3332 ,3323 ,3318 ,3330 ,3323 ,3321 ,3322 ,3332 ,3340 ,3324 ,3323 ,3333 ,3321 ,3317 ,3344 ,
            3339 ,3314 ,3318 ,3328 ,3322 ,3302 ,3303 ,3314 ,3313 ,3320 ,3319 ,3327 ,3297 ,3313 ,3316 ,
            3305 ,3304 ,3293 ,3301 ,3289 ,3312 ,3334 ,3329 ,3321 ,3325 ,3334 ,3329 ,3324 ,3331 ,3323 ,
            3319 ,3324 ,3328 ,3331 ,3328 ,3339 ,3320 ,3323 ,3322 ,3334 ,3340 ,3342 ,3345 ,3323 ,3315 ,
            3305 ,3336 ,3320 ,3321 ,3318 ,3304 ,3314 ,3310 ,3321 ,3308 ,3302 ,3301 ,3300 ,3293 ,3332 ,
            3317 ,3319 ,3313 ,3297 ,3293 ,3302 ,3306 ,3313 ,3308 ,3301 ,3319 ,3318 ,3316 ,3319 ,3317 ,
            3302 ,3306 ,3320 ,3302 ,3320 ,3306 ,3317 ,3308 ,3303 ,3306 ,3298 ,3303 ,3311 ,3295 ,3302 ,
            3303 ,3301 ,3300 ,3299 ,3307 ,3312 ,3321 ,3338 ,3330 ,3320 ,3323 ,3323 ,3315 ,3310 ,3293 ,
            3320 ,3304 ,3331 ,3331 ,3331 ,3312 ,3327 ,3316 ,3303 ,3323 ,3321 ,3318 ,3341 ,3334 ,3318 ,
            3314 ,3302 ,3311 ,3312 ,3311 ,3283 ,3292 ,3317 ,3289 ,3298 ,3291 ,3304 ,3298 ,3301 ,3308 ,
            3312 ,3327 ,3318 ,3310 ,3304 ,3308 ,3315 ,3311 ,3319 ,3326 ,3319 ,3318 ,3309 ,3311 ,3327 ,
            3321 ,3311 ,3317 ,3315 ,3322 ,3302 ,3306 ,3310 ,3308 ,3305 ,3302 ,3305 ,3303 ,3325 ,3296 ,
            3301 ,3296 ,3301 ,3297 ,3317 ,3303 ,3305 ,3309 ,3313 ,3316 ,3304 ,3314 ,3319 ,3334 ,3347 ,
            3330 ,3323 ,3336 ,3341 ,3330 ,3323 ,3315 ,3337 ,3334 ,3333 ,3327 ,3321 ,3315 ,3331 ,3325 ,
            3333 ,3317 ,3314 ,3313 ,3334 ,3321 ,3317 ,3331 ,3314 ,3319 ,3314 ,3305 ,3319 ,3319 ,3306 ,
            3307 ,3292 ,3309 ,3309 ,3300 ,3333 ,3314 ,3316 ,3318 ,3312 ,3324 ,3323 ,3333 ,3325 ,3322 ,
            3331 ,3320 ,3337 ,3328 ,3328 ,3319 ,3314 ,3341 ,3333 ,3317 ,3316 ,3302 ,3305 ,3302 ,3302 ,
            3301 ,3305 ,3312 ,3328 ,3330 ,3316 ,3335 ,3313 ,3317 ,3306 ,3330 ,3325 ,3316 ,3324 ,3306 ,
            3314 ,3328 ,3323 ,3324 ,3323 ,3342 ,3330 ,3322 ,3331 ,3330 ,3313 ,3329 ,3333 ,3338 ,3335 ,
            3309 ,3331 ,3319 ,3313 ,3318 ,3326 ,3313 ,3304 ,3301 ,3304 ,3299 ,3318 ,3323 ,3316 ,3306 ,
            3298 ,3307 ,3310 ,3316 ,3322 ,3315 ,3312 ,3302 ,3310 ,3311 ,3316 ,3305 ,3347 ,3323 ,3305 ,
            3326 ,3327 ,3337 ,3332 ,3336 ,3328 ,3324 ,3330 ,3340 ,3333 ,3318 ,3320 ,3330 ,3333 ,3320 ,
            3305 ,3306 ,3310 ,3312 ,3303 ,3304 ,3317 ,3302 ,3301 ,3312 ,3295 ,3309 ,3330 ,3322 ,3332 ,
            3324 ,3328 ,3347 ,3323 ,3334 ,3330 ,3314 ,3309 ,3311 ,3318 ,3326 ,3318 ,3325 ,3313 ,3311 ,
            3321 ,3323 ,3312 ,3313 ,3313 ,3315 ,3333 ,3322 ,3316 ,3310 ,3321 ,3323 ,3333 ,3316 ,3300 ,
            3303 ,3302 ,3295 ,3304 ,3295 ,3326 ,3334 ,3320 ,3356 ,3340 ,3343 ,3326 ,3315 ,3322 ,3335 ,
            3334 ,3342 ,3334 ,3352 ,3333 ,3325 ,3337 ,3333 ,3344 ,3331 ,3315 ,3330 ,3319 ,3327 ,3317 ,
            3312 ,3312 ,3324 ,3313 ,3321 ,3299 ,3297 ,3304 ,3296 ,3285 ,3305 ,3301 ,3324 ,3324 ,3332 ,
            3322 ,3313 ,3324 ,3338 ,3338 ,3357 ,3342 ,3317 ,3328 ,3334 ,3322 ,3325 ,3317 ,3317 ,3322 ,
            3326 ,3326 ,3329 ,3320 ,3319 ,3331 ,3321 ,3309 ,3322 ,3314 ,3313 ,3307 ,3322 ,3303 ,3315 ,
            3314 ,3314 ,3321 ,3325 ,3325 ,3313 ,3329 ,3356 ,3337 ,3343 ,3327 ,3325 ,3331 ,3337 ,3327 ,
            3325 ,3351 ,3322 ,3329 ,3336 ,3326 ,3307 ,3307 ,3319 ,3320 ,3326 ,3317 ,3309 ,3340 ,3304 ,
            3294 ,3325 ,3331 ,3319 ,3312 ,3301 ,3308 ,3300 ,3301 ,3307 ,3303 ,3312 ,3302 ,3308 ,3296 ,
            3321 ,3299 ,3313 ,3316 ,3317 ,3326 ,3326 ,3322 ,3312 ,3324 ,3319 ,3314 ,3330 ,3324 ,3324 ,
            3328 ,3322 ,3303 ,3315 ,3325 ,3298 ,3315 ,3333 ,3321 ,3328 ,3333 ,3317 ,3322 ,3313 ,3303 ,
            3303 ,3307 ,3323 ,3310 ,3314 ,3293 ,3307 ,3319 ,3318 ,3318 ,3346 ,3333 ,3332 ,3320 ,3324 ,
            3322 ,3320 ,3318 ,3323 ,3313 ,3330 ,3327 ,3324 ,3318 ,3341 ,3325 ,3328 ,3313 ,3303 ,3322 ,
            3314 ,3303 ,3290 ,3302 ,3301 ,3306 ,3312 ,3297 ,3293 ,3296 ,3299 ,3298 ,3279 ,3299 ,3295 ,
            3291 ,3302 ,3311 ,3316 ,3304 ,3317 ,3319 ,3315 ,3322 ,3312 ,3311 ,3326 ,3322 ,3316 ,3314 ,
            3315 ,3304 ,3312 ,3317 ,3304 ,3323 ,3325 ,3321 ,3301 ,3297 ,3308 ,3305 ,3313 ,3302 ,3300 ,
            3294 ,3298 ,3309 ,3303 ,3295 ,3305 ,3298 ,3316 ,3316 ,3327 ,3329 ,3334 ,3340 ,3336 ,3317 ,
            3323 ,3318 ,3336 ,3329 ,3333 ,3314 ,3331 ,3335 ,3326 ,3327 ,3334 ,3327 ,3326 ,3319 ,3323 ,
            3310 ,3317 ,3313 ,3329 ,3332 ,3318 ,3300 ,3311 ,3310 ,3311 ,3305 ,3314 ,3298 ,3303 ,3298 ,
            3309 ,3312 ,3307 ,3314 ,3328 ,3338 ,3334 ,3335 ,3319 ,3337 ,3340 ,3347 ,3331 ,3331 ,3314 ,
            3312 ,3301 ,3313 ,3291 ,3326 ,3319 ,3336 ,3322 ,3310 ,3324 ,3322 ,3313 ,3312 ,3318 ,3319 ,
            3319 ,3304 ,3298 ,3295 ,3297 ,3317 ,3317 ,3297 ,3284 ,3283 ,3296 ,3321 ,3322 ,3322 ,3321 ,
            3325 ,3319 ,3317 ,3301 ,3309 ,3312 ,3309 ,3327 ,3325 ,3310 ,3311 ,3317 ,3322 ,3305 ,3309 ,
            3299 ,3323 ,3304 ,3303 ,3320 ,3313 ,3301 ,3309 ,3288 ,3294 ,3308 ,3308 ,3302 ,3307 ,3310 ,
            3297 ,3296 ,3292 ,3301 ,3308 ,3301 ,3313 ,3312 ,3307 ,3315 ,3316 ,3314 ,3303 ,3308 ,3310 ,
            3314 ,3330 ,3313 ,3299 ,3312 ,3315 ,3319 ,3318 ,3304 ,3310 ,3309 ,3302 ,3312 ,3314 ,3307 ,
            3304 ,3325 ,3310 ,3304 ,3296 ,3295 ,3316 ,3295 ,3301 ,3303 ,3295 ,3316 ,3309 ,3322 ,3310 ,
            3309 ,3327 ,3313 ,3306 ,3319 ,3321 ,3318 ,3327 ,3340 ,3341 ,3324 ,3314 ,3324 ,3326 ,3339 ,
            3322 ,3328 ,3322 ,3319 ,3338 ,3336 ,3324 ,3311 ,3316 ,3320 ,3311 ,3316 ,3316 ,3322 ,3315 ,
            3313 ,3311 ,3290 ,3292 ,3303 ,3302 ,3314 ,3329 ,3324 ,3334 ,3327 ,3325 ,3318 ,3309 ,3315 ,
            3321 ,3311 ,3314 ,3328 ,3329 ,3324 ,3330 ,3341 ,3331 ,3333 ,3336 ,3335 ,3313 ,3308 ,3306 ,
            3322 ,3313 ,3297 ,3311 ,3298 ,3315 ,3307 ,3325 ,3324 ,3315 ,3316 ,3310 ,3309 ,3326 ,3363 ,
            3354 ,3343 ,3346 ,3338 ,3315 ,3330 ,3325 ,3330 ,3331 ,3332 ,3333 ,3326 ,3333 ,3344 ,3328 ,
            3351 ,3352 ,3320 ,3313 ,3324 ,3334 ,3340 ,3323 ,3324 ,3323 ,3300 ,3301 ,3300 ,3294 ,3293 ,
            3297 ,3302 ,3313 ,3309 ,3314 ,3315 ,3319 ,3320 ,3323 ,3324 ,3319 ,3323 ,3323 ,3333 ,3327 ,
            3331 ,3320 ,3334 ,3341 ,3331 ,3336 ,3334 ,3334 ,3334 ,3319 ,3325 ,3324 ,3327 ,3327 ,3335 ,
            3326 ,3335 ,3333 ,3305 ,3309 ,3292 ,3312 ,3312 ,3323 ,3300 ,3300 ,3295 ,3314 ,3309 ,3306 ,
            3309 ,3315 ,3312 ,3319 ,3353 ,3336 ,3337 ,3322 ,3348 ,3327 ,3336 ,3338 ,3332 ,3336 ,3328 ,
            3318 ,3316 ,3326 ,3331 ,3344 ,3328 ,3336 ,3334 ,3339 ,3332 ,3326 ,3330 ,3344 ,3347 ,3342 ,
            3321 ,3323 ,3313 ,3324 ,3322 ,3317 ,3311 ,3319 ,3326 ,3346 ,3333 ,3336 ,3331 ,3322 ,3329 ,
            3324 ,3300 ,3340 ,3345 ,3341 ,3336 ,3329 ,3313 ,3313 ,3321 ,3310 ,3316 ,3340 ,3327 ,3310 ,
            3308 ,3309 ,3303 ,3310 ,3309 ,3318 ,3302 ,3326 ,3308 ,3320 ,3298 ,3301 ,3301 ,3310 ,3308 ,
            3296 ,3284 ,3304 ,3312 ,3306 ,3305 ,3309 ,3311 ,3313 ,3301 ,3308 ,3312 ,3342 ,3332 ,3317 ,
            3317 ,3313 ,3307 ,3318 ,3334 ,3317 ,3310 ,3318 ,3329 ,3321 ,3309 ,3325 ,3316 ,3316 ,3298 ,
            3313 ,3295 ,3315 ,3314 ,3306 ,3327 ,3312 ,3308 ,3315 ,3310 ,3318 ,3324 ,3302 ,3336 ,3317 ,
            3320 ,3321 ,3321 ,3312 ,3314 ,3323 ,3344 ,3349 ,3343 ,3352 ,3365 ,3356 ,3354 ,3335 ,3320 ,
            3318 ,3334 ,3319 ,3320 ,3306 ,3315 ,3317 ,3309 ,3300 ,3298 ,3307 ,3305 ,3296 ,3285 ,3297 ,
            3293 ,3306 ,3319 ,3303 ,3347 ,3333 ,3339 ,3336 ,3318 ,3325 ,3322 ,3334 ,3318 ,3313 ,3345 ,
            3331 ,3331 ,3321 ,3327 ,3323 ,3309 ,3314 ,3315 ,3313 ,3318 ,3321 ,3327 ,3299 ,3307 ,3320 ,
            3312 ,3299 ,3298 ,3295 ,3281 ,3296 ,3283 ,3282 ,3291 ,3278 ,3299 ,3299 ,3292 ,3292 ,3288 ,
            3305 ,3307 ,3316 ,3306 ,3309 ,3312 ,3304 ,3318 ,3317 ,3322 ,3325 ,3328 ,3329 ,3314 ,3304 ,
            3305 ,3320 ,3333 ,3322 ,3334 ,3330 ,3310 ,3300 ,3309 ,3297 ,3307 ,3304 ,3301 ,3307 ,3302 ,
            3312 ,3307 ,3300 ,3293 ,3291 ,3311 ,3333 ,3332 ,3329 ,3312 ,3327 ,3311 ,3312 ,3314 ,3311 ,
            3324 ,3303 ,3323 ,3321 ,3325 ,3338 ,3335 ,3311 ,3331 ,3315 ,3308 ,3310 ,3307 ,3312 ,3319 ,
            3304 ,3309 ,3304 ,3295 ,3302 ,3295 ,3287 ,3285 ,3288 ,3295 ,3280 ,3301 ,3295 ,3302 ,3321 ,
            3319 ,3317 ,3303 ,3308 ,3336 ,3337 ,3329 ,3339 ,3345 ,3324 ,3327 ,3320 ,3329 ,3325 ,3345 ,
            3332 ,3351 ,3349 ,3330 ,3334 ,3352 ,3350 ,3332 ,3353 ,3352 ,3352 ,3342 ,3337 ,3323 ,3312 ,
            3327 ,3314 ,3311 ,3314 ,3312 ,3312 ,3329 ,3332 ,3332 ,3335 ,3350 ,3342 ,3335 ,3328 ,3323 ,
            3326 ,3323 ,3322 ,3319 ,3322 ,3322 ,3327 ,3352 ,3320 ,3321 ,3328 ,3333 ,3317 ,3326 ,3313 ,
            3316 ,3302 ,3313 ,3308 ,3298 ,3302 ,3316 ,3315 ,3309 ,3299 ,3314 ,3300 ,3299 ,3303 ,3287 ,
            3312 ,3293 ,3312 ,3303 ,3318 ,3317 ,3313 ,3295 ,3296 ,3300 ,3298 ,3307 ,3309 ,3318 ,3318 ,
            3315 ,3310 ,3313 ,3324 ,3319 ,3312 ,3317 ,3320 ,3316 ,3303 ,3313 ,3312 ,3309 ,3315 ,3310 ,
            3301 ,3295 ,3303 ,3307 ,3299 ,3308 ,3308 ,3313 ,3297 ,3321 ,3304 ,3301 ,3293 ,3305 ,3302 ,
            3312 ,3315 ,3321 ,3322 ,3325 ,3314 ,3300 ,3317 ,3309 ,3315 ,3320 ,3307 ,3299 ,3306 ,3305 ,
            3305 ,3290 ,3304 ,3282 ,3290 ,3278 ,3284 ,3275 ,3280 ,3287 ,3273 ,3274 ,3279 ,3292 ,3289 ,
            3280 ,3289 ,3293 ,3316 ,3307 ,3293 ,3303 ,3304 ,3315 ,3293 ,3306 ,3314 ,3311 ,3305 ,3308 ,
            3307 ,3320 ,3316 ,3309 ,3306 ,3302 ,3301 ,3323 ,3311 ,3302 ,3312 ,3287 ,3289 ,3286 ,3285 ,
            3294 ,3302 ,3288 ,3307 ,3291 ,3286 ,3286 ,3290 ,3294 ,3300 ,3302 ,3312 ,3313 ,3318 ,3320 ,
            3329 ,3327 ,3311 ,3322 ,3322 ,3326 ,3337 ,3324 ,3326 ,3315 ,3313 ,3330 ,3330 ,3320 ,3327 ,
            3312 ,3327 ,3329 ,3319 ,3297 ,3295 ,3267 ,3289 ,3284 ,3297 ,3299 ,3285 ,3288 ,3294 ,3293 ,
            3283 ,3301 ,3302 ,3286 ,3303 ,3302 ,3309 ,3319 ,3314 ,3318 ,3304 ,3307 ,3292 ,3303 ,3310 ,
            3308 ,3323 ,3315 ,3305 ,3316 ,3309 ,3326 ,3318 ,3321 ,3297 ,3323 ,3337 ,3341 ,3321 ,3303 ,
            3303 ,3311 ,3311 ,3304 ,3300 ,3293 ,3305 ,3288 ,3306 ,3300 ,3303 ,3294 ,3290 ,3307 ,3297 ,
            3312 ,3300 ,3299 ,3315 ,3315 ,3316 ,3305 ,3311 ,3304 ,3298 ,3305 ,3295 ,3313 ,3311 ,3307 ,
            3303 ,3304 ,3312 ,3300 ,3338 ,3304 ,3319 ,3312 ,3291 ,3300 ,3291 ,3289 ,3293 ,3298 ,3296 ,
            3303 ,3296 ,3303 ,3295 ,3302 ,3304 ,3303 ,3307 ,3311 ,3302 ,3304 ,3313 ,3315 ,3311 ,3324 ,
            3323 ,3341 ,3313 ,3319 ,3319 ,3320 ,3295 ,3302 ,3306 ,3316 ,3322 ,3309 ,3296 ,3314 ,3298 ,
            3302 ,3307 ,3303 ,3310 ,3313 ,3310 ,3314 ,3314 ,3291 ,3298 ,3307 ,3306 ,3300 ,3296 ,3309 ,
            3303 ,3311 ,3298 ,3298 ,3313 ,3315 ,3313 ,3304 ,3316 ,3300 ,3313 ,3313 ,3300 ,3326 ,3316 ,
            3319 ,3314 ,3321 ,3307 ,3301 ,3320 ,3320 ,3305 ,3297 ,3296 ,3292 ,3297 ,3286 ,3291 ,3297 ,
            3287 ,3282 ,3302 ,3312 ,3310 ,3291 ,3308 ,3320 ,3347 ,3323 ,3321 ,3317 ,3331 ,3336 ,3324 ,
            3322 ,3334 ,3338 ,3348 ,3344 ,3341 ,3340 ,3338 ,3345 ,3338 ,3332 ,3343 ,3339 ,3331 ,3325 ,
            3339 ,3327 ,3320 ,3316 ,3330 ,3307 ,3301 ,3312 ,3323 ,3304 ,3305 ,3302 ,3288 ,3296 ,3291 ,
            3317 ,3321 ,3333 ,3328 ,3339 ,3326 ,3317 ,3318 ,3309 ,3306 ,3308 ,3312 ,3316 ,3300 ,3310 ,
            3314 ,3312 ,3285 ,3309 ,3323 ,3328 ,3325 ,3327 ,3323 ,3315 ,3315 ,3318 ,3311 ,3299 ,3295 ,
            3313 ,3299 ,3308 ,3329 ,3308 ,3307 ,3291 ,3303 ,3313 ,3300 ,3310 ,3307 ,3319 ,3312 ,3324 ,
            3325 ,3313 ,3324 ,3318 ,3323 ,3310 ,3309 ,3313 ,3325 ,3317 ,3333 ,3332 ,3325 ,3310 ,3320 ,
            3337 ,3320 ,3319 ,3325 ,3309 ,3307 ,3317 ,3324 ,3314 ,3315 ,3311 ,3317 ,3306 ,3307 ,3286 ,
            3301 ,3299 ,3304 ,3321 ,3309 ,3300 ,3300 ,3302 ,3320 ,3321 ,3333 ,3320 ,3314 ,3321 ,3318 ,
            3313 ,3318 ,3314 ,3330 ,3326 ,3325 ,3325 ,3315 ,3307 ,3319 ,3317 ,3304 ,3308 ,3313 ,3305 ,
            3298 ,3306 ,3309 ,3320 ,3303 ,3290 ,3291 ,3301 ,3298 ,3307 ,3310 ,3305 ,3303 ,3319 ,3328 ,
            3325 ,3303 ,3311 ,3299 ,3302 ,3302 ,3302 ,3314 ,3305 ,3327 ,3315 ,3316 ,3326 ,3320 ,3325 ,
            3325 ,3326 ,3309 ,3311 ,3306 ,3309 ,3315 ,3308 ,3309 ,3301 ,3316 ,3301 ,3288 ,3298 ,3313 ,
            3318 ,3313 ,3318 ,3332 ,3311 ,3312 ,3321 ,3319 ,3327 ,3319 ,3319 ,3321 ,3340 ,3326 ,3322 ,
            3339 ,3337 ,3319 ,3350 ,3329 ,3336 ,3350 ,3352 ,3343 ,3348 ,3322 ,3321 ,3321 ,3339 ,3314 ,
            3297 ,3302 ,3307 ,3299 ,3305 ,3293 ,3296 ,3305 ,3303 ,3315 ,3303 ,3311 ,3302 ,3311 ,3306 ,
            3318 ,3314 ,3309 ,3298 ,3302 ,3318 ,3308 ,3312 ,3315 ,3323 ,3330 ,3296 ,3309 ,3307 ,3311 ,
            3332 ,3311 ,3309 ,3325 ,3328 ,3339 ,3305 ,3326 ,3314 ,3306 ,3300 ,3308 ,3302 ,3298 ,3309 ,
            3294 ,3301 ,3308 ,3309 ,3321 ,3314 ,3333 ,3342 ,3339 ,3335 ,3327 ,3309 ,3330 ,3346 ,3329 ,
            3328 ,3320 ,3322 ,3327 ,3307 ,3329 ,3333 ,3330 ,3340 ,3337 ,3328 ,3340 ,3341 ,3336 ,3308 ,
            3309 ,3300 ,3315 ,3323 ,3308 ,3303 ,3308 ,3310 ,3292 ,3289 ,3267 ,3286 ,3303 ,3323 ,3314 ,
            3315 ,3326 ,3315 ,3319 ,3313 ,3314 ,3318 ,3331 ,3325 ,3308 ,3317 ,3325 ,3309 ,3312 ,3335 ,
            3319 ,3320 ,3321 ,3321 ,3327 ,3333 ,3325 ,3326 ,3331 ,3326 ,3323 ,3324 ,3312 ,3305 ,3309 ,
            3297 ,3292 ,3308 ,3297 ,3295 ,3300 ,3297 ,3314 ,3312 ,3307 ,3316 ,3312 ,3331 ,3329 ,3338 ,
            3329 ,3334 ,3317 ,3325 ,3330 ,3340 ,3358 ,3357 ,3359 ,3330 ,3331 ,3341 ,3331 ,3318 ,3327 ,
            3327 ,3309 ,3303 ,3327 ,3321 ,3313 ,3308 ,3314 ,3293 ,3299 ,3306 ,3323 ,3304 ,3297 ,3315 ,
            3314 ,3308 ,3323 ,3335 ,3346 ,3341 ,3331 ,3314 ,3331 ,3321 ,3328 ,3328 ,3326 ,3322 ,3329 ,
            3322 ,3316 ,3318 ,3342 ,3324 ,3317 ,3318 ,3317 ,3315 ,3306 ,3303 ,3319 ,3325 ,3319 ,3327 ,
            3308 ,3324 ,3318 ,3317 ,3304 ,3319 ,3305 ,3302 ,3321 ,3323 ,3319 ,3304 ,3311 ,3311 ,3324 ,
            3324 ,3316 ,3332 ,3337 ,3317 ,3320 ,3328 ,3332 ,3324 ,3311 ,3300 ,3304 ,3305 ,3316 ,3320 ,
            3313 ,3324 ,3306 ,3314 ,3308 ,3299 ,3305 ,3304 ,3317 ,3315 ,3304 ,3311 ,3319 ,3312 ,3327 ,
            3347 ,3324 ,3328 ,3324 ,3315 ,3328 ,3323 ,3319 ,3331 ,3329 ,3319 ,3314 ,3328 ,3324 ,3330 ,
            3340 ,3358 ,3336 ,3356 ,3348 ,3343 ,3353 ,3329 ,3334 ,3324 ,3316 ,3335 ,3322 ,3309 ,3318 ,
            3299 ,3320 ,3311 ,3308 ,3325 ,3317 ,3317 ,3321 ,3319 ,3321 ,3319 ,3325 ,3326 ,3323 ,3317 ,
            3318 ,3299 ,3313 ,3328 ,3319 ,3322 ,3330 ,3316 ,3310 ,3306 ,3309 ,3309 ,3310 ,3315 ,3315 ,
            3310 ,3308 ,3306 ,3305 ,3310 ,3311 ,3302 ,3308 ,3320 ,3317 ,3289 ,3302 ,3312 ,3299 ,3306 ,
            3305 ,3300 ,3329 ,3315 ,3324 ,3332 ,3316 ,3330 ,3318 ,3309 ,3316 ,3328 ,3334 ,3320 ,3335 ,
            3329 ,3328 ,3318 ,3318 ,3329 ,3318 ,3321 ,3323 ,3318 ,3326 ,3320 ,3317 ,3320 ,3336 ,3330 ,
            3312 ,3318 ,3313 ,3322 ,3328 ,3323 ,3329 ,3321 ,3331 ,3329 ,3315 ,3318 ,3335 ,3339 ,3323 ,
            3309 ,3320 ,3327 ,3340 ,3323 ,3325 ,3330 ,3326 ,3327 ,3330 ,3323 ,3331 ,3318 ,3320 ,3320 ,
            3335 ,3316 ,3330 ,3317 ,3296 ,3302 ,3299 ,3299 ,3299 ,3297 ,3305 ,3312 ,3299 ,3289 ,3281 ,
            3299 ,3296 ,3311 ,3301 ,3321 ,3306 ,3307 ,3325 ,3313 ,3323 ,3322 ,3315 ,3321 ,3323 ,3305 ,
            3317 ,3319 ,3320 ,3331 ,3320 ,3322 ,3319 ,3319 ,3322 ,3324 ,3315 ,3311 ,3306 ,3287 ,3291 ,
            3295 ,3291 ,3300 ,3287 ,3303 ,3303 ,3318 ,3308 ,3302 ,3294 ,3308 ,3295 ,3303 ,3318 ,3340 ,
            3332 ,3317 ,3336 ,3323 ,3307 ,3328 ,3329 ,3334 ,3347 ,3320 ,3331 ,3323 ,3319 ,3324 ,3327 ,
            3318 ,3334 ,3316 ,3337 ,3348 ,3334 ,3327 ,3311 ,3304 ,3309 ,3304 ,3294 ,3291 ,3301 ,3308 ,
            3302 ,3295 ,3300 ,3301 ,3299 ,3313 ,3313 ,3324 ,3306 ,3296 ,3313 ,3307 ,3316 ,3311 ,3316 ,
            3317 ,3312 ,3307 ,3304 ,3306 ,3317 ,3323 ,3331 ,3322 ,3327 ,3320 ,3323 ,3323 ,3311 ,3314 ,
            3310 ,3312 ,3309 ,3323 ,3303 ,3305 ,3294 ,3315 ,3291 ,3307 ,3313 ,3304 ,3312 ,3295 ,3298 ,
            3309 ,3320 ,3316 ,3324 ,3326 ,3323 ,3329 ,3296 ,3316 ,3325 ,3311 ,3328 ,3336 ,3318 ,3334 ,
            3326 ,3319 ,3333 ,3312 ,3310 ,3308 ,3317 ,3304 ,3315 ,3292 ,3305 ,3308 ,3302 ,3301 ,3302 ,
            3319 ,3304 ,3313 ,3312 ,3314 ,3305 ,3320 ,3317 ,3323 ,3314 ,3312 ,3337 ,3327 ,3311 ,3311 ,
            3315 ,3311 ,3316 ,3303 ,3310 ,3327 ,3318 ,3341 ,3344 ,3324 ,3336 ,3344 ,3330 ,3330 ,3335 ,
            3328 ,3320 ,3329 ,3322 ,3321 ,3322 ,3303 ,3315 ,3307 ,3303 ,3298 ,3301 ,3304 ,3298 ,3321 ,
            3293 ,3310 ,3320 ,3319 ,3313 ,3303 ,3325 ,3328 ,3323 ,3319 ,3317 ,3314 ,3319 ,3326 ,3309 ,
            3328 ,3311 ,3330 ,3317 ,3305 ,3316 ,3308 ,3318 ,3326 ,3327 ,3325 ,3322 ,3320 ,3318 ,3318 ,
            3315 ,3300 ,3306 ,3301 ,3306 ,3290 ,3305 ,3294 ,3300 ,3302 ,3310 ,3320 ,3325 ,3326 ,3310 ,
            3318 ,3322 ,3314 ,3327 ,3318 ,3314 ,3315 ,3308 ,3315 ,3306 ,3304 ,3310 ,3297 ,3301 ,3321 ,
            3320 ,3314 ,3323 ,3320 ,3332 ,3315 ,3304 ,3303 ,3315 ,3306 ,3313 ,3311 ,3309 ,3296 ,3276 ,
            3284 ,3288 ,3303 ,3300 ,3315 ,3311 ,3316 ,3317 ,3325 ,3320 ,3332 ,3327 ,3314 ,3321 ,3334 ,
            3331 ,3328 ,3299 ,3320 ,3317 ,3310 ,3316 ,3318 ,3313 ,3299 ,3317 ,3310 ,3301 ,3307 ,3320 ,
            3298 ,3310 ,3313 ,3304 ,3296 ,3310 ,3294 ,3310 ,3309 ,3301 ,3312 ,3309 ,3324 ,3328 ,3319 ,
            3318 ,3308 ,3317 ,3319 ,3316 ,3314 ,3323 ,3332 ,3321 ,3309 ,3328 ,3313 ,3330 ,3329 ,3315 ,
            3335 ,3328 ,3321 ,3320 ,3327 ,3307 ,3320 ,3331 ,3312 ,3324 ,3316 ,3311 ,3315 ,3286 ,3314 ,
            3302 ,3297 ,3292 ,3285 ,3313 ,3314 ,3328 ,3322 ,3322 ,3327 ,3328 ,3321 ,3331 ,3349 ,3334 ,
            3348 ,3341 ,3320 ,3318 ,3325 ,3323 ,3311 ,3323 ,3334 ,3337 ,3329 ,3322 ,3308 ,3303 ,3303 ,
            3312 ,3309 ,3299 ,3295 ,3289 ,3307 ,3307 ,3319 ,3307 ,3306 ,3299 ,3308 ,3315 ,3326 ,3320 ,
            3307 ,3299 ,3312 ,3304 ,3315 ,3302 ,3321 ,3330 ,3338 ,3337 ,3328 ,3324 ,3323 ,3315 ,3314 ,
            3323 ,3323 ,3347 ,3337 ,3335 ,3330 ,3328 ,3317 ,3316 ,3324 ,3318 ,3326 ,3317 ,3299 ,3326 ,
            3314 ,3311 ,3310 ,3310 ,3305 ,3323 ,3315 ,3326 ,3341 ,3333 ,3324 ,3326 ,3329 ,3339 ,3328 ,
            3321 ,3331 ,3334 ,3325 ,3309 ,3316 ,3328 ,3331 ,3330 ,3341 ,3315 ,3322 ,3324 ,3319 ,3311 ,
            3309 ,3321 ,3304 ,3311 ,3317 ,3342 ,3336 ,3331 ,3309 ,3304 ,3309 ,3297 ,3297 ,3293 ,3302 ,
            3310 ,3313 ,3314 ,3319 ,3312 ,3318 ,3322 ,3316 ,3317 ,3324 ,3328 ,3323 ,3317 ,3317 ,3308 ,
            3311 ,3321 ,3323 ,3309 ,3307 ,3310 ,3324 ,3313 ,3311 ,3311 ,3306 ,3319 ,3306 ,3301 ,3288 ,
            3299 ,3301 ,3301 ,3299 ,3320 ,3297 ,3329 ,3319 ,3313 ,3325 ,3330 ,3328 ,3331 ,3323 ,3329 ,
            3324 ,3328 ,3315 ,3323 ,3328 ,3317 ,3326 ,3306 ,3320 ,3310 ,3309 ,3311 ,3313 ,3319 ,3310 ,
            3316 ,3327 ,3316 ,3299 ,3311 ,3283 ,3282 ,3295 ,3302 ,3293 ,3299 ,3288 ,3288 ,3294 ,3311 ,
            3308 ,3316 ,3309 ,3309 ,3320 ,3316 ,3330 ,3325 ,3332 ,3325 ,3324 ,3322 ,3323 ,3332 ,3349 ,
            3347 ,3335 ,3338 ,3321 ,3337 ,3305 ,3301 ,3314 ,3309 ,3318 ,3300 ,3306 ,3320 ,3317 ,3306 ,
            3307 ,3292 ,3286 ,3294 ,3309 ,3299 ,3311 ,3323 ,3327 ,3310 ,3313 ,3318 ,3322 ,3325 ,3329 ,
            3318 ,3318 ,3322 ,3315 ,3316 ,3298 ,3302 ,3309 ,3320 ,3316 ,3317 ,3313 ,3315 ,3306 ,3325 ,
            3326 ,3326 ,3306 ,3298 ,3295 ,3290 ,3303 ,3296 ,3304 ,3308 ,3310 ,3296 ,3299 ,3323 ,3315 ,
            3320 ,3314 ,3332 ,3327 ,3323 ,3339 ,3348 ,3337 ,3317 ,3312 ,3313 ,3322 ,3329 ,3329 ,3319 ,
            3316 ,3325 ,3318 ,3319 ,3323 ,3319 ,3321 ,3327 ,3330 ,3345 ,3327 ,3310 ,3314 ,3321 ,3304 ,
            3305 ,3308 ,3326 ,3310 ,3315 ,3314 ,3302 ,3313 ,3307 ,3307 ,3326 ,3321 ,3341 ,3341 ,3345 ,
            3338 ,3350 ,3338 ,3335 ,3325 ,3338 ,3327 ,3301 ,3321 ,3319 ,3337 ,3327 ,3324 ,3327 ,3329 ,
            3338 ,3349 ,3324 ,3328 ,3324 ,3321 ,3307 ,3313 ,3311 ,3325 ,3344 ,3344 ,3345 ,3330 ,3313 ,
            3305 ,3310 ,3330 ,3320 ,3320 ,3344 ,3353 ,3339 ,3320 ,3312 ,3310 ,3317 ,3321 ,3335 ,3336 ,
            3343 ,3315 ,3315 ,3314 ,3323 ,3327 ,3327 ,3342 ,3327 ,3325 ,3359 ,3337 ,3312 ,3323 ,3323 ,
            3322 ,3326 ,3312 ,3321 ,3323 ,3309 ,3316 ,3315 ,3300 ,3298 ,3298 ,3297 ,3305 ,3329 ,3323 ,
            3329 ,3316 ,3312 ,3302 ,3301 ,3312 ,3310 ,3306 ,3308 ,3305 ,3320 ,3321 ,3324 ,3323 ,3323 ,
            3318 ,3314 ,3305 ,3320 ,3312 ,3313 ,3317 ,3302 ,3305 ,3305 ,3305 ,3311 ,3294 ,3295 ,3309 ,
            3303 ,3296 ,3310 ,3287 ,3301 ,3306 ,3306 ,3310 ,3321 ,3322 ,3337 ,3319 ,3305 ,3304 ,3307 ,
            3313 ,3320 ,3313 ,3330 ,3325 ,3329 ,3333 ,3322 ,3335 ,3331 ,3326 ,3325 ,3311 ,3328 ,3309 ,
            3312 ,3312 ,3319 ,3308 ,3327 ,3316 ,3323 ,3312 ,3304 ,3298 ,3292 ,3305 ,3301 ,3323 ,3313 ,
            3309 ,3320 ,3302 ,3303 ,3336 ,3335 ,3319 ,3328 ,3332 ,3333 ,3326 ,3342 ,3321 ,3327 ,3355 ,
            3322 ,3315 ,3322 ,3319 ,3334 ,3347 ,3333 ,3334 ,3331 ,3313]
    data = np.array(data,dtype="float")
    ###---set up the time-series.  4590FPS.
    inttime = 202e-6
    
    data_clean = [i for i in data if (not np.isnan(i))]
    ts_clean = np.arange(0,len(data_clean)/4590,1/4590)    #times, units of seconds
    
    all_ts = np.arange(0,len(data)/4590,1/4590)    #times, units of seconds
    ts_nan = []
    for k in range(len(data)):
        if not np.isnan(data[k]):
            ts_nan.append(all_ts[k])
    ts_nan = np.array(ts_nan,dtype=ts_clean.dtype)
    
    # =============================================================================
    # #time-series plot
    # fig_t,ax_t = plt.subplots()
    # ax_t.plot(ts_clean,data_clean,"o-b",label="raw data (nans removed)")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Raw digital counts")
    # ax_t.set_title("Raw time-series")
    # 
    # fig_t,ax_t = plt.subplots()
    # ax_t.plot(ts_nan,data_clean,"o-b",label="raw data (nans present)")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Raw digital counts")
    # ax_t.set_title("Raw time-series, nans ommitted")
    # =============================================================================
    
    
    
    
    
    ###---------TESTING---------------------------------------------------------###

    ###---build the time-series
    fps = 100    #frames per second
    num = 2000  #number of data points
    ts = np.arange(0,num/fps,1/fps)    #times, units of seconds
    ts = np.linspace(0,num/fps,num,endpoint=False)   #times, units of seconds
    f1 = 8      #frequencies of the sine wave signal
    f2 = 21     #frequencies of the sine wave signal
    a1 = 1      #amplitude of the 1st sine wave frequency
    a2 = 2      #amplitude of the 2nd sine wave frequency
    dc = 8      #overal offset
    Noise = lambda t: a1*np.sin(2*np.pi*f1*t) + a2*np.sin(2*np.pi*f2*t)+dc
    sines = Noise(ts)  #data output



    def TestTransforms(ys,fps,logscaleY=False):
        """Test whether the standard fourier transform, the hanning fourier
        transform, and the non-uniform fourier transform all give the same
        results on uniform data"""
        ts = np.arange(0,len(ys)/fps,1/fps)    #times, units of seconds
        
        fig_f,ax_f = plt.subplots()
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("components")
        if logscaleY:
            ax_f.set_yscale("log")
    
        ###---plot standard fourier transform
        fs,fourier = StandardFourier(ys,fps)
        ax_f.plot(fs, fourier, "o-b",label="standard fft")
        ax_f.legend()
        print(np.sum(fourier))
        
        ###---plot standard fourier transform
        smoothed = HanningApply(ys,show=False)
        fs2,fourier2 = StandardFourier(smoothed,fps)
        ax_f.plot(fs2, fourier2, "og",label="standard fft with hanning")
        ax_f.legend()
        print(np.sum(fourier2))
        
        ###---plot slow version of the non-uniform discrete fourier transform
        fs3,fourier3 = SlowNonUniform(times=ts,frames=ys,fps=fps)
        ax_f.plot(fs3, fourier3, ".r",label="slow nufft algorithm")
        ax_f.legend()
        ax_f.set_title("Testing different fft methods for consistency")
    
    
    
    
    def TestHalfRate(ts,ys,fps):
        """test having half-rate data"""
        fig_h,ax_h = plt.subplots()
        ax_h.set_xlabel("Frequency (Hz)")
        ax_h.set_ylabel("Components of the sine waves")
        ax_h.set_title("Testing different FPS and durations")
        #time-series plot
        fig_t,ax_t = plt.subplots()
        ax_t.plot(ts,ys,"o-b",label="raw data")
        ax_t.set_xlabel("time (s)")
        ax_t.set_ylabel("Current of some sort")
        ax_t.set_title("Time-series with different data rates")
        
        ###---plot standard fourier transform
        fs,fourier = StandardFourier(ys,fps)
        ax_h.plot(fs, fourier, "-",color='cyan',label="reference")
        ax_h.legend()
        
        ###---plot transform of same data but at half-FPS
        tslow = ts[::2]+(ts[1]-ts[0])/2
        slow = Noise(tslow)
        ax_t.plot(tslow,slow,".r",label="half FPS")
        ax_t.legend()
        fs5,fourier5 = StandardFourier(slow,fps/2)
        ax_h.plot(fs5, fourier5, "or",label="half FPS")
        ax_h.legend()
        ax_h.set_title("Fourier components of different data rates")
        
    
    
    # TestHalfRate(ts,sines,fps)
    # TestTransforms(sines,fps)
    # TestTransforms(data_clean,4590,logscaleY=True)
    
    
    
    
    
    
    # ###---repeat the same tests as above, except this time plotting spectral density
    # ###---test having half-rate or half-duration data
    # ys = sines
    # fig_h,ax_h = plt.subplots()
    # ax_h.set_xlabel("Frequency (Hz)")
    # ax_h.set_ylabel("RMS_units / sqrt(Hz)")
    # ax_h.set_title("Testing ComponentsToSpectralDensity")
    # #time-series plot
    # fig_t,ax_t = plt.subplots()
    # ax_t.plot(ts,ys,"o-b",label="raw data")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Current of some sort")
    # ax_t.set_title("Testing ComponentsToSpectralDensity")
    
    # ###---plot standard fourier transform
    # freq_ref,four_ref = StandardFourier(ys,fps)
    # y_show = ComponentsToSpectralDensity(freq_ref,four_ref)
    # ax_h.plot(freq_ref,y_show, "-o",color='cyan',label="reference")
    # ax_h.legend()
    
    # ###---plot transform of same data but at half-FPS
    # tslow = ts[::2]+(ts[1]-ts[0])/2
    # slow = Noise(tslow)
    # ax_t.plot(tslow,slow,".r",label="half FPS")
    # ax_t.legend()
    # freq_slow,four_slow = StandardFourier(slow,fps/2)
    # y_show = ComponentsToSpectralDensity(freq_slow,four_slow)
    # ax_h.plot(freq_slow, y_show, "or",label="half FPS")
    # ax_h.legend()
    
    # ###---plot transform of same data but at half-duration of the data run
    # halfdata = ys[:len(ys)//2]
    # ax_t.plot(ts[:len(ts)//2],halfdata,".g",label="half duration")
    # ax_t.legend()
    # freq_half,four_half = StandardFourier(halfdata,fps)
    # y_show = ComponentsToSpectralDensity(freq_half,four_half)
    # ax_h.plot(freq_half,y_show, ".g",label="half duration") #(low by sqrt2 b/c Hz bins twice as wide)
    # ax_h.legend()
    
    
    
    ###---does our normal transform work on snippets of the same data?  If I break the data into two parts?
    def TestSnippets(ys,fps,ratio,logscaleY=True,showsegments=True):
        """Given time-series data "ys" taken at the frame rate "fps" (in Hz), breaks
        the series into two snippets (the first being "ratio" fraction of the
        time-series and the second being 1-"ratio") and compares the fourier
        transform of the whole to the fourier transform of each snippet"""
        fig_f,ax_f = plt.subplots()
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("components")
        if logscaleY:
            ax_f.set_yscale("log")
            
        #plot standard fourier transform
        freqs,fours = StandardFourier(ys,fps)
        ax_f.plot(freqs,fours, ".-b",label="fft of whole data run")
        ax_f.legend()
        
        #find fourier of partial data
        assert(ratio<1 and ratio>0)
        index = int(ratio*len(ys))
        freqs1,fours1 = StandardFourier(ys[:index],fps)
        freqs2,fours2 = StandardFourier(ys[index:],fps)
        
        #show fourier of partial data
        if showsegments:
            ax_f.plot(freqs1,fours1, ".r",label="fft of first %0.1f%% of data run" %(ratio*100))
            ax_f.plot(freqs2,fours2, ".",color="orange",label="fft of final %0.1f%% of data run" %((1-ratio)*100))
    
        #try to interpolate the two partial-sections back to the full frequency spacing
        interp1 = InterpFourierComponents(freqs,freqs1,fours1)
        ax_f.plot(freqs,interp1, "s",markersize=4,markerfacecolor="none",markeredgecolor="red",label="interp from first %0.1f%% of data" %(ratio*100))
        interp2 = InterpFourierComponents(freqs,freqs2,fours2)
        ax_f.plot(freqs,interp2, "s",markersize=7,markerfacecolor="none",markeredgecolor="orange",label="interp from final %0.1f%% of data" %((1-ratio)*100))
        
        #average the two partial-sections back together
        rejoined = np.average([interp1,interp2],axis=0)  #arithmetic average
        # rejoined = np.power(10,np.average([np.log10(interp1),np.log10(interp2)],axis=0)) #geometric average
        ax_f.plot(freqs,rejoined, ".-k",label="interpolations averaged")
        
        
        
        ax_f.legend()
        print("\n")
        print("sum of fourier of total:",np.sum(fours))
        print("sum of fourier of first:",np.sum(fours1))
        print("sum of interpl of first:",np.sum(interp1))
        print("sum of fourier of final:",np.sum(fours2))
        print("sum of interpl of final:",np.sum(interp2))
        print("average of both interps:",np.sum(rejoined))
    
    
    
    # #testing with sine waves
    # TestSnippets(sines,fps=100,ratio = 1/7)
    
    # #testing with noisy data
    # TestSnippets(data_clean,fps=4590,ratio = 2/7,showsegments=False)
    
    
    
    
    
    
    
            
    
    
    # =============================================================================
    # #testing using sine wave data -- works!
    # sines_nan = sines.copy()
    # sines_nan[50:110] = np.nan
    # sines_nan[1718:1800] = np.nan
    # 
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(sines_nan)/100,1/100) #fps=100
    # ax_t.plot(ts,sines,"o-g",label="real data set")
    # ax_t.plot(ts,sines_nan,".r",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Sine wave data set")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = fig_f,ax_f = plt.subplots()
    # freq,four = StandardFourier(HanningApply(sines),fps=100)
    # four = np.where(four<1e-3,0,four)
    # ax.plot(freq[2:],four[2:],"o-g",label="true full frequency")
    # FourierAvoidingNanSegments(sines_nan,fps=100,show=True,usehann=True,weighavrg=True,ax=ax,spectraldensity=False)
    # ax.set_title("Segment test for sine wave data")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # 
    # 
    # #testing using continuous section of real noise data -- works!
    # data_clean_nan = data[130:].copy()
    # data_clean_nan[2300:2500] = np.nan
    # data_clean_nan[3000:] = np.nan
    # 
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(data_clean_nan)/4590,1/4590) #fps=4590
    # ax_t.plot(ts,data[130:],"o-g",label="real data set")
    # ax_t.plot(ts,data_clean_nan,".r",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Non-skipping real noise data")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = fig_f,ax_f = plt.subplots()
    # freq,four = StandardFourier(data[130:],fps=4590)
    # ax.plot(freq[2:],four[2:],"o-g",label="true full frequency")
    # FourierAvoidingNanSegments(data_clean_nan,fps=4590,show=True,usehann=False,weighavrg=True,ax=ax,spectraldensity=False)
    # ax.set_title("Segment test for real noise data (no real skips, only artificial)")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # 
    # 
    # #testing using real noise data -- works!
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(data)/4590,1/4590) #fps=4590
    # ax_t.plot(ts,data,"-or",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Real noise data with observed real frame skips")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = plt.subplots()
    # FourierAvoidingNanSegments(data,fps=4590,show=True,usehann=True,weighavrg=True,ax=ax,spectraldensity=False)
    # ax.set_title("Segment test for real noise data")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # 
    # =============================================================================
    
    
    # =============================================================================
    # ###---now I just need to convert to spectral density and rerun the above three tests
    # 
    # ###---testing using sine wave data -- works!
    # sines_nan = sines.copy()
    # sines_nan[50:110] = np.nan
    # sines_nan[1718:1800] = np.nan
    # 
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(sines_nan)/100,1/100) #fps=100
    # ax_t.plot(ts,sines,"o-g",label="real data set")
    # ax_t.plot(ts,sines_nan,".r",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Sine wave data set")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = fig_f,ax_f = plt.subplots()
    # freq,four = StandardFourier(HanningApply(sines),fps=100)
    # four = np.where(four<1e-3,0,four)
    # density = ComponentsToSpectralDensity(freq,four)
    # ax.plot(freq[2:],density[2:],"o-g",label="true full frequency")
    # FourierAvoidingNanSegments(sines_nan,fps=100,show=True,usehann=True,weighavrg=True,ax=ax,spectraldensity=True)
    # ax.set_title("Segment test for sine wave data")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # 
    # ###---testing using continuous section of real noise data -- works!
    # data_clean_nan = data[130:].copy()
    # data_clean_nan[2300:2500] = np.nan
    # data_clean_nan[3000:] = np.nan
    # 
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(data_clean_nan)/4590,1/4590) #fps=4590
    # ax_t.plot(ts,data[130:],"o-g",label="real data set")
    # ax_t.plot(ts,data_clean_nan,".r",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Non-skipping real noise data")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = fig_f,ax_f = plt.subplots()
    # freq,four = StandardFourier(data[130:],fps=4590)
    # density = ComponentsToSpectralDensity(freq,four)
    # ax.plot(freq[2:],density[2:],"o-g",label="true full frequency")
    # FourierAvoidingNanSegments(data_clean_nan,fps=4590,show=True,usehann=False,weighavrg=True,ax=ax,spectraldensity=True)
    # ax.set_title("Segment test for real noise data (no real skips, only artificial)")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # 
    # ###---testing using real noise data -- works!
    # fig_t,ax_t = plt.subplots()
    # ts = np.arange(0,len(data)/4590,1/4590) #fps=4590
    # ax_t.plot(ts,data,"-or",label="skipping frames")
    # ax_t.set_xlabel("time (s)")
    # ax_t.set_ylabel("Data counts")
    # ax_t.set_title("Real noise data with observed real frame skips")
    # ax_t.legend()
    # fig_t.tight_layout()
    # 
    # fig,ax = plt.subplots()
    # data_current = data * 10.37597 * 7.93168e-16 #CountstoE(1,0,9809)*EtoI(1,202e-6)
    # FourierAvoidingNanSegments(data_current,fps=4590,show=True,usehann=True,weighavrg=True,ax=ax,spectraldensity=True)
    # ax.set_title("Segment test for real noise data")
    # # ax.set_ylim(bottom=2e-9)
    # fig.tight_layout()
    # =============================================================================
    
    
    
    ###---so: working procedure is:
    ###---freqs,spectraldensity = FourierAvoidingNanSegments(data,fps,show,usehann=True,weighavrg=True,ax=None,spectraldensity=True)
        