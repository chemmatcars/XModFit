subroutine ff_cylinder(q,R,L,ff,M)
    !***************************************************************************
    !Subroutine to calculate the form factor of cylinder
    !q = Array of reciprocal wave-vectors at which the form-factor needs to be calculated
    !R = Radius of cylinder in Angstroms
    !L = Length of cylinder in Angstroms
    !M = No. of reciprocal wave-vectors at which the form-factor needs to be calculated
    !***************************************************************************
    integer :: M, i, j
    double precision :: q(0:M)
    double precision :: ff(0:M)
    double precision :: R, L, alpha, fact
    double precision, parameter :: pi=3.14157
    integer, parameter :: N=1000

    do i = 0,M
        do j=1,N
            alpha=pi*j/N/2.0
            fact=(sin(q(i)*L*cos(alpha)/2)/(q(i)*L*cos(alpha)/2) * bessel_j1(q(i)*R*sin(alpha))/(q(i)*R*sin(alpha)))**2
            ff(i) = ff(i) + fact * sin(alpha)
        enddo
        ff(i) = ff(i)*pi/N
    enddo

end subroutine ff_cylinder


subroutine ff_cylinder_dist(q,R,Rdist,L,Ldist,ff,M,N)
    !***************************************************************************
    !Subroutine to calculate the form factor of cylinder with disctribution of radius and length
    !q = Array of reciprocal wave-vectors at which the form-factor needs to be calculated
    !R = Array of radii of cylinder in Angstroms
    !Rdist = Probablity distribution of radii R
    !L = Array of Length of cylinder in Angstroms
    !Ldist = Probablity distribution of length L
    !N = No. of samples R and L
    !ff = Form factor of cylinder
    !M = No. of reciprocal wave-vectors at which the form-factor needs to be calculated
    !***************************************************************************
    integer :: M, N, i, j
    double precision :: q(0:M)
    double precision :: ff(0:M)
    double precision :: R(0:N), Rdist(0:N), L(0:N), Ldist(0:N), fact1, fact2, tsum
    double precision, parameter :: pi=3.14157
    integer, parameter :: Nangles=1000
    double precision :: alpha(0:Nangles)

    do j=1,Nangles
        alpha(j)=pi*j/Nangles/2.0
    end do

    do i = 0,M
        ff(i)=0.0d0
        do k=0,N
            tsum=0.0d0
            do j=1,Nangles
                fact1=sin(q(i)*L(k)*cos(alpha(j))/2.0d0)/(q(i)*L(k)*cos(alpha(j))/2.0d0)
                fact2=bessel_j1(q(i)*R(k)*sin(alpha(j)))/(q(i)*R(k)*sin(alpha(j)))
                tsum = tsum + fact1**2*fact2**2*sin(alpha(j))
            enddo
            ff(i) = ff(i)+tsum*Rdist(k)*Ldist(k)
        end do
        ff(i) = ff(i)*pi/sum(Rdist)/sum(Ldist)/Nangles
    enddo

end subroutine ff_cylinder_dist


subroutine ff_cylinder_ml(q,rho,R,L,ff,ffamp,M, Nl)
    !***************************************************************************
    !Subroutine to calculate the form factor of Multilayered Cylinder
    !q = Array of reciprocal wave-vectors at which the form-factor needs to be calculated
    !rho= List of complex electron density of different layers including the solvent
    !R = List of Radius and shell thicknesses of multilayered cylinder in Angstroms
    !L = Length of cylinder in Angstroms
    !M = No. of reciprocal wave-vectors at which the form-factor needs to be calculated
    !Nl = No. of multilayers
    !***************************************************************************
    integer :: M, Nl, i, j, K
    double precision :: q(0:M)
    double precision :: ff(0:M)
    double complex :: rho(0:Nl)
    double precision :: R(0:Nl)
    double precision :: L, alpha, rtemp
    double complex :: fact
    double complex :: ffamp(0:M)
    double precision, parameter :: pi=3.14157
    integer, parameter :: N=1000

    do i = 0,M
        do j=1,N
            alpha=pi*j/N/2.0
            fact=dcmplx(0.0,0.0)
            rtemp=0
            do k=0,Nl-1
                rtemp=rtemp+R(k)
                fact=fact+(rho(k+1)-rho(k))*(dsin(q(i)*L*dcos(alpha)/2)/(q(i)*L*dcos(alpha)/2) &
                * bessel_j1(q(i)*rtemp*dsin(alpha))/(q(i)*rtemp*dsin(alpha)))
            enddo
            ff(i) = ff(i) + cdabs(fact)**2 * dsin(alpha)
            ffamp(i) = ffamp(i) + fact * dsin(alpha)
        enddo
        ff(i) = ff(i)*pi/N
        ffamp(i) = ffamp(i)*pi/N
    enddo

end subroutine ff_cylinder_ml

subroutine ff_cylinder_ml_asaxs(q,H,R,rho,eirho,adensity,Nalf,fft,ffs,ffc,ffr,M,Nlayers)
    !***************************************************************************
    !Subroutine to calculate the form factor of multilayered cylinder with disctribution of radii
    !q = Array of reciprocal wave-vectors at which the form-factor needs to be calculated
    !H = Height of the cylinder
    !R = Array of half of the radii of the core and shells of the cylinder
    !rho = Array of energy dependent electron densities  of core and the shells
    !eirho = Array of energy indepdent electron densities of core and the shells
    !adensity = Array of density of the resonant element in the particle
    !Nlayers = Number of shells plus the core
    !Nalf = Number of azimuthal angle to be integrated for isotropic integration
    !fft = Total scattering from the ellipsoid particle
    !ffs = SAXS-term
    !ffc = Cross-term
    !ffr = Resonant term
    !M = No. of reciprocal wave-vectors at which the form-factor needs to be calculated
    !Nlayers = No. of multilayers in the cylinder
    !***************************************************************************
    integer :: M, i, j, k, Nalf, Nlayers
    double precision :: q(0:M-1)
    double precision :: fft(0:M-1), ffs(0:M-1), ffc(0:M-1), ffr(0:M-1)
    double precision :: R(0:Nlayers-1)
    double complex :: rho(0:Nlayers-1)
    double precision :: eirho(0:Nlayers-1), adensity(0:Nlayers-1)
    double precision :: H, fs, fc, fr, tR, alf, dalf, V, fac
    double complex :: ft, tft
    double precision :: tfs, tfr
    dalf=3.14159/float(Nalf)
    do i = 0,M-1
        fft(i)=0.0d0
        ffs(i)=0.0d0
        ffc(i)=0.0d0
        ffr(i)=0.0d0
        do j = 1, Nalf
            alf=float(j)*dalf
            tR=0.0d0
            tft=0.0d0
            tfs=0.0d0
            tfr=0.0d0
            do k = 0, Nlayers-2
                tR=tR+R(k)
                V=3.14159*tR**2*H
                fac=(dsin(q(i)*H*dcos(alf)/2)/(q(i)*H*dcos(alf)/2) &
                * bessel_j1(q(i)*tR*dsin(alf))/(q(i)*tR*dsin(alf)))
                ft=2*V*(rho(k)-rho(k+1))*fac
                fs=2*V*(eirho(k)-eirho(k+1))*fac
                fr=2*V*(adensity(k)-adensity(k+1))*fac
                fc=fs*fr
                tft=tft+ft
                tfs=tfs+fs
                tfr=tfr+fr
            end do
            fft(i)=fft(i)+cdabs(tft)**2*dsin(alf)
            ffs(i)=ffs(i)+tfs**2*dsin(alf)
            ffc(i)=ffc(i)+tfs*tfr*dsin(alf)
            ffr(i)=ffr(i)+tfr**2*dsin(alf)
        end do

        fft(i)=fft(i)*dalf
        ffs(i)=ffs(i)*dalf
        ffc(i)=ffc(i)*dalf
        ffr(i)=ffr(i)*dalf
    end do

end subroutine ff_cylinder_ml_asaxs
