# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:42:25 2021

@author: Dr David Anderson

Description:
    This code simulates an extremely simple 2D model of an omnicopter. 
    Simulation is achieved using the omniCopter class shown below, which 
    contains the model parameters and a function to calculate the derivative
    vector of the omnicopter states. There are 8 states used to describe the 
    motion of the omnicopter and these are:
        x = [y,yd,z,zd,phi,phid,phird,phild]^T
        where:
            y = lateral position
            yd = lateral position rate
            z = vertical position
            zd = vertical position rate
            phi = roll angle
            phid = roll rate
            phir = right rotor roll angle
            phil = left rotor roll angle
            
    The model also accepts the 4 control actions which you will use machine 
    learning methods to find. These are:
        act = [Tr,phidr,Tl,phild]
        where:
            Tr = right rotor thrust
            phird = right rotor roll angle rate
            Tl = left rotor thrust
            phild = left rotor roll angle rate
            
    This python script has been developed to allow you to see the code required
    to run a single event. This should be packaged appropriately into your own
    machine learning script.
"""
import numpy as np
import matplotlib.pyplot as plt
import math

class omniCopter():
    def __init__(self):
        self.running = True
        self.m = 1
        self.Ixx = 0.2
        self.l = 0.2
        
    def calcDerivatives(self,x,act,xd):
        # Extract the actions
        Tr = act[0]
        phird = act[1]
        Tl = act[2]
        phild = act[3]
        # Calculate the rotor forces in earth axes
        # Create the tilting rotor direction cosines
        phir = x[6]
        phil = x[7]
        Cpr_b = np.array([[1,0,0],
                          [0,math.cos(phir),-math.sin(phir)],
                          [0,math.sin(phir),math.cos(phir)]])
        Cpl_b = np.array([[1,0,0],
                          [0,math.cos(phil),-math.sin(phil)],
                          [0,math.sin(phil),math.cos(phil)]])
        Tvr = np.array([[0.0],[0.0],-Tr],dtype=object)
        Tvl = np.array([[0.0],[0.0],-Tl],dtype=object)
        Fr_b = Cpr_b @ Tvr
        Fl_b = Cpl_b @ Tvl
        # Now the body to NED axes
        phi = x[4]
        Cb_e = np.array([[1,0,0],
                          [0,math.cos(phi),-math.sin(phi)],
                          [0,math.sin(phi),math.cos(phi)]])
        # Then,
        Fr_e = Cb_e @ Fr_b
        Fl_e = Cb_e @ Fl_b
        # Total forces acting on the body are then,
        g = 10
        F = Fr_e + Fl_e + Cb_e @ np.array([[0],[0],[self.m * g]],dtype=object)
        # Now the moments. First transgform the moment arms into NED axes
        r_cg_pr_e = Cb_e @ np.array([[0],[self.l],[0]],dtype=object)
        r_cg_pl_e = Cb_e @ np.array([[0],[-self.l],[0]],dtype=object)
        # Now calculate the torque vector
        Tq = np.cross(np.transpose(r_cg_pr_e),np.transpose(Fr_e)) \
           + np.cross(np.transpose(r_cg_pl_e),np.transpose(Fl_e)) 
        #
        # With the forces and moments found, we can compute the linear and 
        # angular accelerations.
        ydd = F[1][0] / self.m
        zdd = F[2][0] / self.m
        phidd = Tq[0][1] / self.Ixx
        # Return the derivative vectors
        xd[0] = x[1]
        xd[1] = ydd
        xd[2] = x[3]
        xd[3] = zdd
        xd[4] = x[5]
        xd[5] = phidd
        xd[6] = phird
        xd[7] = phild
        return xd
        
    def runningStatus(self,s):
        self.running = s
        return self.running
            
            
if __name__ == "__main__":
    # Create an omnicopter object
    oc = omniCopter()
    
    # Define the environment 
    yp = 0.0
    zp = -1.0
    phip = 0.0
    ylim = [-5, 5]
    zlim = [-10, 0]
    
    # Initialise the actions. 
    act = np.array([[0.0],[0.6],[-5.0],[0.6]])
    
    # Set the animation running (TBC)
    running  = True
    
    # Create container lists to hold the plot data
    yplt = np.array([])
    zplt = []
    phiplt = []
    tplt = []
    
    # Initialise the state vector
    y0 = -0.0
    z0 = -8.0
    phi0 = 0.0
    x = np.array([[y0],[0.0],[z0],[0.0],[phi0],[0.0],[0.0],[0.0]])
    xd = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    # Set the event end time
    tf = 10      # sec
    t = np.array([0.0])
    while running:
        # Calculate the actions. 
        #
        # ACTIONS FROM YOUR MACHINE LEARNING CODE GO IN HERE
        #
        
        # Integrate the ODEs
        dt = np.array([0.01])
        xd = np.zeros((8,1))
        xd = oc.calcDerivatives(x,act,xd)
        # Euler integration
        x += xd*dt
        t += dt
        # Store data for plotting
        yplt = np.append(yplt,x[0])
        zplt = np.append(zplt,-x[2])
        phiplt = np.append(phiplt,x[4])
        tplt = np.append(tplt,t)
        # Check if within bounds
        valid = (x[0][0] > ylim[0]) and (x[0][0] < ylim[1]) \
            and (x[2][0] > zlim[0]) and (x[2][0] < zlim[1])
        # print("time = ",t,"\t z = ",x[2][0],"valid = ",valid)
        # Check while loop status
        running = oc.runningStatus(valid and t<tf)
        
    # Plot the data
    plt.figure(figsize=(4,3),dpi=300)
    plt.plot(yplt,zplt,label='omnicopter')
    plt.xlabel(r'$y$-axis')
    plt.ylabel(r'$-z$-axis')
    plt.axis([-5,5,0,10])
    plt.show()