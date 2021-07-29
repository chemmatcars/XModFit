f2py -c --compiler=mingw32 ff_cylinder.pyf ff_cylinder.f90
f2py -c --compiler=mingw32 ff_sphere.pyf ff_sphere.f90
f2py -c --compiler=mingw32 ff_ellipsoid.pyf ff_ellipsoid.f90
f2py -c --compiler=mingw32 xr_ref.pyf xr_ref.f90
f2py -c --compiler=mingw32 TwoD_Integrate.pyf TwoD_Integrate.f90
f2py -c --compiler=mingw32 -m ff_parallelepiped ff_parallelepiped.f90