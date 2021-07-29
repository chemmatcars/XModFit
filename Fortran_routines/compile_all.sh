f2py -c ff_cylinder.pyf ff_cylinder.f90
f2py -c ff_sphere.pyf ff_sphere.f90
f2py -c ff_ellipsoid.pyf ff_ellipsoid.f90
f2py -c xr_ref.pyf xr_ref.f90
f2py -c TwoD_Integrate.pyf TwoD_Integrate.f90
f2py -c -m ff_parallelepiped ff_parallelepiped.f90
