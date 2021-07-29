subroutine ff_parallelepiped(q, a, b, L, N, ff, M)
    !***************************************************************************
    !Subroutine to calculate the form factor of parallelepiped
    !q = Array of reciprocal wave-vectors at which the form-factor needs to be calculated
    !a = 1st dimension of the cross-section in Angstroms
    !b = 2nd dimension of the cross-section in Angstroms
    !L = length of the parallelepiped in Angstroms
    !M = No. of reciprocal wave-vectors at which the form-factor needs to be calculated
    !N = No. of points at which the discrete angular integration will be done
    !***************************************************************************
    integer :: M, i, j
    double precision, intent(in) :: q(0:M-1)
    double precision, intent(out) :: ff(0:M-1)
    double precision, intent(in) :: a
    double precision, intent(in) :: b
    double precision, intent(in) :: L
    integer, intent(in):: N
    double precision :: phi, psi
    double precision, parameter :: pi=3.14157d0
    double precision :: aa, bb, ll

    do i = 0,M
        ff(i) = 0.0d0
        do j = 1, N
            phi = j * 2 * pi / N
            do k = 1, N
                psi = k * pi / N
                aa = q(i) * a * dsin(phi) * dcos(psi / 2.0d0)
                bb = q(i) * b * dsin(phi) * dsin(psi / 2.0d0)
                ll = q(i) * L * dcos(phi / 2.0d0)
                ff(i)=ff(i) + (dsin(aa) * dsin(bb) * dsin(ll))**2 / (aa * bb * ll)**2
            end do
        enddo
        ff(i) = (a*b*L)**2 * ff(i) * pi / N**2 / 4
    enddo

end subroutine ff_parallelepiped


