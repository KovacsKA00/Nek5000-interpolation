import os
import pymech as pm
import numpy as np
import fnmatch
from numba import jit
import vtk
import pyvista as pv


# Case parameters
ldomain = 0.05 # The same in each coordinate direction
nodes_per_element_in_x_dir = 16 # GLL points

# Interpolation functions
def compute_lagrange_basis(x):
    n = len(x)
    basis_coefficients = []

    for i in range(n):
        # Start with a constant polynomial '1'
        p_coeff = np.array([1])
        for j in range(n):
            if i != j:
                # Multiply by the (x - x[j]) term and divide by (x[i] - x[j])
                factor = np.array([-x[j], 1])
                p_coeff = np.convolve(p_coeff, factor) / (x[i] - x[j])
        basis_coefficients.append(p_coeff)

    # Ensure all polynomials have the same length
    basis_coefficients = [np.pad(coeff, (0, n - len(coeff)), 'constant') for coeff in basis_coefficients]

    return np.array(basis_coefficients)

@jit(nopython=True)
def evaluate_polynomial(coeffs, x):
    result = 0.0
    for c in coeffs[::-1]:
        result = result * x + c
    return result

@jit(nopython=True)
def interpolate_with_basis(x_new, u, basis_coefficients):
    n = len(u)
    interpolated = np.zeros(x_new.shape, dtype=x_new.dtype)

    for i in range(n):
        # Evaluate each Lagrange basis polynomial at points in x_new
        for j in range(len(x_new)):
            interpolated[j] += evaluate_polynomial(basis_coefficients[i], x_new[j]) * u[i]

    return interpolated

def interpolate_nek_field(np_field, x, nodes_per_element_in_x_dir, double_yn=True):
    # Interpolation from Gauss-Lobato-Lagrange (GLL) points into equidistant grid
    # Domain size and resolution are the same in the x-y-z directions!
    # If double_yn is set to True then the resolution will be doubled
    nn = nodes_per_element_in_x_dir
    ne = int(x.shape[0] / nn)
    l_domain = x[-1]

    x_gll = x[:nn]                 
    basis_polynomials = compute_lagrange_basis(x_gll)

    if double_yn:
        step = 2
        nn = nn * 2
        u_new = np.tile(np.zeros_like(np_field), (2, 2, 2))
        u_new[::2, ::2, ::2] = np_field.copy()
    else:
        step = 1
        u_new = np_field.copy()

    dx = l_domain / (nn * ne)
    x_lin = np.linspace(0, (nn * dx) - dx, nn)

    for i in range(ne):
        for j in range(0, nn * ne, step):
            for k in range(0, nn * ne, step):
                u_new[k, j, i*nn:(i+1)*nn] = interpolate_with_basis(x_lin, u_new[k, j, i*nn:(i+1)*nn:step], basis_polynomials)

    for j in range(ne):
        for i in range(nn * ne):
            for k in range(0, nn * ne, step):
                u_new[k, j*nn:(j+1)*nn, i] = interpolate_with_basis(x_lin, u_new[k, j*nn:(j+1)*nn:step, i], basis_polynomials)

    for k in range(ne):
        k_range = np.arange(k*nn, (k+1)*nn)
        for i in range(nn * ne):
            for j in range(nn * ne):
                u_new[k*nn:(k+1)*nn, j, i] = interpolate_with_basis(x_lin, u_new[k*nn:(k+1)*nn:step, j, i], basis_polynomials)
    
    return u_new
# Interpolation functions END

def list_files_with_filter(data_folder, filter_str):
    # List file names matching the filter string
    files = os.listdir(data_folder)
    filtered_files = [file for file in files if fnmatch.fnmatch(file, filter_str)]
    return filtered_files


def interpolate_velocity_field(ds, nodes_per_element_in_x_dir):
    # Interpolate velocity data from GLL to equidistant grid
    vel = np.stack([interpolate_nek_field(ds.ux.to_numpy(), ds.x.to_numpy(), nodes_per_element_in_x_dir),
                interpolate_nek_field(ds.uy.to_numpy(), ds.x.to_numpy(), nodes_per_element_in_x_dir), 
                interpolate_nek_field(ds.uz.to_numpy(), ds.x.to_numpy(), nodes_per_element_in_x_dir)])
    vel = np.transpose(vel, (3, 2, 1, 0))
    return vel

# List the names of the cases
projects = ['Case1', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6', 'Case7', 'Case8', 'Case9', 'Case10',
            'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case16', 'Case17', 'Case18', 'Case19']

n = 256 # Grid resolution
x = np.linspace(0,0.05,n)
y = np.linspace(0,0.05,n)
z = np.linspace(0,0.05,n)

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

for i, project in enumerate(projects):
    # Filenames: write your folder name here
    data_folder = './DNS_Kolmogorov/' + project + '/'

    # Process all NEK output files in the data_folder
    f_names = list_files_with_filter(data_folder, 'shear_3d0.f*')

    for f_name in f_names[0:]:
        print(f_name)
        ds = pm.open_dataset(data_folder + f_name)
        vel = interpolate_velocity_field(ds, nodes_per_element_in_x_dir)
        print(np.shape(vel))
        velocity_vectors = np.stack([vel[:,:,:,0].ravel(), vel[:,:,:,1].ravel(), vel[:,:,:,2].ravel()], axis=-1)
        print(np.shape(velocity_vectors))
        grid = pv.StructuredGrid(Z, Y, X)
        grid["Velocity"] = velocity_vectors
        grid.save(data_folder + "vel_"+f_name[-4:]+".vtk")