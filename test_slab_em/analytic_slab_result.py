""" """

from scipy.special import wofz, iv
import numpy as np
import matplotlib.pyplot as plt

def Gammafunc(arg):
    """ """
    return np.exp(-arg)*iv(0,arg)

def plasmadispfunc(arg):
    """ """
    return wofz(arg)*1j

def test_plasmadispfunc():
    """ """
    print("plasmadispfunc(1e-5) = ", plasmadispfunc(1e-5))
    print("plasmadispfunc(0.01) = ", plasmadispfunc(0.01))
    print("plasmadispfunc(1) = ", plasmadispfunc(1))
    print("plasmadispfunc(100) = ", plasmadispfunc(100))

class SlabDispersionRelation:

    def __init__(self, Ti_norm, Te_norm, mi_norm, me_norm, B_norm, beta, kpa_norm, kperp_norm):
        """ """
        self.Ti_norm = Ti_norm
        self.Te_norm = Te_norm
        self.mi_norm = mi_norm
        self.me_norm = me_norm
        self.B_norm = B_norm
        self.beta = beta
        self.kpa_norm = kpa_norm
        self.kperp_norm = kperp_norm

        ## Now get the derived values
        self.vthi_norm = (self.Ti_norm/self.mi_norm)**0.5 # Thermal velocity for ions. Normalised to vthr = sqrt(2Tr/mr)
        self.vthe_norm = (self.Te_norm/self.me_norm)**0.5 # Thermal velocity for electrons. Normalised to vthr = sqrt(2Tr/mr)
        self.Omegai_norm = self.B_norm/self.mi_norm # Gyrofrequency for ions. Normalised: Omegai = Omegai_norm*e*Br/(mr*c)
        self.Omegae_norm = self.B_norm/self.me_norm # Gyrofrequency for electrons. Normalised: Omegai = Omegai_norm*e*Br/(mr*c)
        self.rhoi_norm = self.vthi_norm/self.Omegai_norm # Ion Larmor radius. Normalised: rhoi = rhoi_norm * (vthr/Omegar)
        self.rhoe_norm = self.vthe_norm/self.Omegae_norm # Electron normalised Larmor radius
        self.bi_norm = self.kperp_norm*self.kperp_norm*self.rhoi_norm*self.rhoi_norm/2 # Arugment of the Gamma function for ions
        self.be_norm = self.kperp_norm*self.kperp_norm*self.rhoe_norm*self.rhoe_norm/2 # Arugment of the Gamma function for ions
        self.Gammai = Gammafunc(self.bi_norm)
        self.Gammae = Gammafunc(self.be_norm)
        return

    def dispersion_relation(self, omega):
        """ """
        omegabari = omega/(self.kpa_norm*self.vthi_norm)
        omegabare = omega/(self.kpa_norm*self.vthe_norm)
        dispi = plasmadispfunc(omegabari)
        dispe = plasmadispfunc(omegabare)
        # kappa =
        # A0 = -kappa/(self.kperp_norm*self.kperp_norm-omega/(kpa*c)*kappa)
        return

    def get_dispersion_relation(self):
        """ """


        return


if __name__ =="__main__":
    print("Hello world")
    my_disp = SlabDispersionRelation(1, 1, 1, 2.7e-4, 1, 0.01, 1, 0.1)
    # print("self.rhoi_norm = ", my_disp.rhoi_norm)
    # print("self.rhoe_norm = ", my_disp.rhoe_norm)
    # print("self.bi_norm = ", my_disp.bi_norm)
    # print("self.be_norm = ", my_disp.be_norm)
    # print("self.Gammai = ", my_disp.Gammai)
    # print("self.Gammae = ", my_disp.Gammae)
