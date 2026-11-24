import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

sinogram = cv2.imread("walnut_divergent.png", cv2.IMREAD_GRAYSCALE)
print(sinogram.shape)


def rebinning(sinogram, fod, fdd, W, center_offset):
    num_beta, num_gamma = sinogram.shape
    reconstruction = np.zeros((num_beta, num_gamma))

    Gamma = 2 * np.arctan(W / (2*fdd)) 

    beta_range = np.linspace(-Gamma / 2, 2 * np.pi + Gamma / 2, num_beta, endpoint=False)
    gamma_range = np.linspace(-Gamma / 2, Gamma / 2, num_gamma)

    theta = np.linspace(0, 2 * np.pi, num_beta, endpoint=False)
    s_max = fod * np.sin(Gamma / 2)
    s = np.linspace(-s_max, s_max, num_gamma)

    for i in range(num_gamma):
        gamma = - np.arcsin(s[i]/fod)
        if gamma < - Gamma / 2:
            gamma += Gamma / 2
        elif gamma >= Gamma / 2:
            gamma -= Gamma / 2

        for j in range(num_beta):
            beta = theta[j] - gamma

            index_gamma = np.argmin(np.abs(gamma_range - gamma))
            index_beta = np.argmin(np.abs(beta_range - beta))

            if beta < -Gamma / 2:
                beta += 2 * np.pi + Gamma / 2
            elif beta >= 2 * np.pi + Gamma / 2:
                beta -= 2 * np.pi + Gamma / 2
            
            reconstruction[j, i] = sinogram[index_beta, index_gamma]

    return reconstruction

parallel_sino = rebinning(sinogram, 540, 630, 120, 0)
print(parallel_sino.shape)

# num_theta = parallel_sino.shape[1]
# theta = np.linspace(0, 360, num_theta, endpoint=False)
# reconstructed_image = iradon(parallel_sino, theta)
# cv2.imshow('Recovered Image', reconstructed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def ram_lak_filter(size):
    freqs = np.fft.fftfreq(size).reshape(-1, 1)
    #print("freqs=",freqs)
    filter = 1-2*np.abs(freqs)
    #print("filter=",filter)
    return filter

def CTRadon(sinogram, angle):
    num_angles, num_projections = sinogram.shape
    angle_diff = np.linspace(0, angle, num_angles)
    
    res = np.zeros((num_projections, num_projections))
    
    center = (num_projections // 2, num_projections // 2)

    for num_angle in range(num_angles):
        x_ray = sinogram[num_angle, :]

        # Fourier 1D
        x_ray_fourier = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_ray)))
        
        # Filter
        filter = ram_lak_filter(num_projections)
        x_ray_fourier_filtered = x_ray_fourier * filter.squeeze()
        
        # Inverse Fourier
        x_ray_back = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x_ray_fourier_filtered))).real
        
        swift = np.tile(x_ray_back, (num_projections, 1))  # Repeat the x_ray to form an image
        matrix = cv2.getRotationMatrix2D(center=center, angle=angle_diff[num_angle], scale=1)
        rotated = cv2.warpAffine(swift, matrix, (num_projections, num_projections))
        
        res += rotated
    
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title('CT Reconstructie')
    plt.show()
    print("Size of picture",res.shape)
# Voer de CTRadon functie uit met het geladen sinogram
CTRadon(parallel_sino, 360)
