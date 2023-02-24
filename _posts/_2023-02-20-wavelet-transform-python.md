The continuous wavelet transform (CWT) equation is:
$$
C(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t)\Psi^*\left(\frac{t-b}{a}\right)dt
$$
where $x(t)$ is the input signal, $\Psi^*(t)$ is the complex conjugate of the mother wavelet function $\Psi(t)$, and $a$ and $b$ are the scale and translation parameters, respectively.

The discrete wavelet transform (DWT) equation is:
$$
c_{j,k} = \langle x,\psi_{j,k}\rangle = \frac{1}{\sqrt{2^j}}\int_{-\infty}^\infty x(t) \psi^*_{j,k}(t)dt
$$
where $x$ is the input signal, $\psi_{j,k}$ is the wavelet function at scale $j$ and position $k$, and $c_{j,k}$ is the wavelet coefficient at scale $j$ and position $k$. The wavelet function is obtained by scaling and translating the mother wavelet function $\psi(t)$, such that $\psi_{j,k}(t) = 2^{-j/2} \psi\left(2^{-j}t - k\right)$. The wavelet coefficients can be computed using a filter bank approach.

Note that there are different wavelet families, such as Daubechies, Coiflets, and Symlets, that have different properties and are used for different applications.