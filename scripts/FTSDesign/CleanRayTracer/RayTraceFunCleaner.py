#This is a cleaned up version of RayTraceFunClean.py. Mira
import numpy as np
import numpy
from random import uniform
import random

'''Rotations: Give angle wanted rotated to respective function, returns rotated point(s).'''

def Rx(x):
    Rx = np.matrix([[1,0,0],[0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return Rx
def Ry(y):
    Ry = np.matrix([[np.cos(y),0,np.sin(y)],[0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return Ry
def Rz(z):
    Rz = np.matrix([[np.cos(z), - np.sin(z), 0],[np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz
def Rxyz (thet):
    Rxyz = Rx(thet[0])*Ry(thet[1])*Rz(thet[2])
    return Rxyz

#Functions for Making the Center Ellipse(s)

''' EllipseLineInt(ELI): Give point of the line, vector of the line, and coefficients of the ellipse, find the intersection(s) of the line and the ellipsoid (assuming ellipse is rotated about the x-axis. '''
def ELI(pli,v1,coeffellipse):
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[1]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[1]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[1]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[1]**2))
    C3 = (pli[2]**2)/(coeffellipse[1]**2) + (pli[2]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli[0])/(v1[0]))*v1[1] + pli[1],((xint[1] - pli[0])/(v1[0]))*v1[1] + pli[1]]
    zint = [((xint[0] - pli[0])/(v1[0]))*v1[2] + pli[2], ((xint[1] - pli[0])/(v1[0]))*v1[2] + pli[2]]
    return xint,yint,zint 



#incorrect: 
'''def ELI(pli,v1,coeffellipse):
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[0]**2))
    B1 = -(2*pli[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    B2 = (2*v1[1]*pli[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli[2])/((v1[0])*coeffellipse[0]**2)
    B = B1 + B2
    C1 = ((pli[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    C2 = -(2*pli[0]*v1[1]*pli[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli[0]*v1[2]*pli[2])/((v1[0])*(coeffellipse[0]**2))
    C3 = (pli[2]**2)/(coeffellipse[0]**2) + (pli[1]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli[0])/(v1[0]))*v1[1] + pli[1],((xint[1] - pli[0])/(v1[0]))*v1[1] + pli[1]]
    zint = [((xint[0] - pli[0])/(v1[0]))*v1[2] + pli[2], ((xint[1] - pli[0])/(v1[0]))*v1[2] + pli[2]]
    return xint,yint,zint'''

'''TangCoef (TC): give point of intersection of an ellipse, returns the coefficients of the tangent plane (for BOTH intersection points) '''
def TC(pli,v1,coeffellipse):
    xint1, yint1, zint1 = ELI(pli,v1,coeffellipse)
    cpos = [xint1[0]/(coeffellipse[0]**2),yint1[0]/(coeffellipse[1]**2),zint1[0]/(coeffellipse[0]**2),1]
    cneg = [xint1[1]/(coeffellipse[0]**2),yint1[1]/(coeffellipse[1]**2),zint1[1]/(coeffellipse[0]**2),1]
    c = [cpos, cneg]
    return c

def NormalP(pli,v1,coeffellipse):
    xint1, yint1, zint1 = ELI(pli,v1,coeffellipse)
    cpos = [(2*xint1[0])/(coeffellipse[0]**2),(2*yint1[0])/(coeffellipse[1]**2),(2*zint1[0])/(coeffellipse[1]**2)]
    cneg = [(2*xint1[1])/(coeffellipse[0]**2),(2*yint1[1])/(coeffellipse[1]**2),(2*zint1[1])/(coeffellipse[1]**2)]
    cpos = np.array(cpos)
    cneg = np.array(cneg)
    return cpos,cneg

''' norm(N): Given a vector, returns the normal of it''' 
def N(V):
    VectL = numpy.array(V)
    VNorm = numpy.sqrt(VectL[0]**2 + VectL[1]**2 + VectL[2]**2)
    VectLNorm = ([u/VNorm for u in VectL])
    VectLNorm = numpy.array(VectLNorm)
    return VectLNorm

'''make_line (ML): given a point, vector, and length, makes the corresponding line '''
def ML(p,v,L):
    pointL = p
    VectL = numpy.array(v)
    Lwant = int(L)
    VectLNorm = N(v)
    t = numpy.linspace(0,Lwant,50) #make related to wanted length??
    x = [pointL[0]]
    y = [pointL[1]]
    z = [pointL[2]]
    for t in range (0,Lwant):
        L = numpy.sqrt(((VectLNorm[0]*t)**2 + (VectLNorm[2]*t)**2 + (VectLNorm[2]*t)**2))
        xL = pointL[0] + t*VectLNorm[0]
        yL = pointL[1] + t*VectLNorm[1]
        zL = pointL[2] + t*VectLNorm[2]
        if L <= Lwant:
            x.append(xL)
            y.append(yL)
            z.append(zL)
    return x,y,z

''' plane_info (PI): Given the coefficients of a plane, get the normal. '''
def PI(C):
    coeff1 = numpy.array(C)
    r = int(100)
    xpl,ypl,zpl = numpy.meshgrid(range(r), range(r), range(r))
    U = coeff1[0]*xpl + coeff1[1]*ypl + coeff1[2]*zpl
    dU = coeff1[:-1] #gradient (for plane)
    #normalized normal line
    Nor = N(dU)
    return dU, Nor

'''make_plane(MP): given coefficients of plane, returns the plane. to be used for a surface plot.'''
def MP(C):
    coeff1 = numpy.array(C)
    r = int(100)
    xp, yp = numpy.meshgrid(range(-r,r), range(-r,r))
    zp = (-coeff1[0]*xp - coeff1[1]*yp + coeff1[3])/coeff1[2]
    return xp,yp,zp

'''ReflectEllipsePoint (REP): Give coefficients of ellipse, vector of the initial ray and the source of the initial ray. Returns the intial ray, reflected ray, plane coefficients, and intersections for both positive and negative. 
def REP(coeffellipse,v,p): #p = origin is origin of source! (source origin)
    c = TC(p,v,coeffellipse) #plane coefficients
    cpos = c[0] #coefficients of positive intersection
    cneg = c[1] #coefficients of negative intersection
    VectLinit = v #initial vector
    VectLNorm = N(v) #incident unit vector
    dU, Npos = PI(cpos) #gradient and normal of plane of positive intersection
    dUneg, Nneg = PI(cneg) #gradient and normal of plane of negative intersection
    VectL2 = VectLNorm - 2*Npos #reflected vector of positive intersection
    #VectLNorm2 = N(VectL2) #reflected unit vector of positive intersection
    VectL2neg = VectLNorm - 2*Nneg
    #VectLNorm2neg = N(VectL2neg)
    #xp,yp,zp = MP(cpos) #plane of positive intersection
    #xp,yp,zp = MP(cneg) #plane of negative intersection
    xint,yint,zint = ELI(p,v,coeffellipse)
    pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    pointintneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    L1 = np.sqrt((pointint[0] - p[0])**2 + (pointint[1] - p[1])**2 + (pointint[2] - p[2])**2)
    L1neg = np.sqrt((pointintneg[0] - p[0])**2 + (pointintneg[1] - p[1])**2 + (pointintneg[2] - p[2])**2)
    xi,yi,zi = ML(p,VectLinit,L1) #incident line from positive intersection point
    xr,yr,zr = ML(pointint,VectL2,L1) #reflective line from positive intersection point
    xin,yin,zin = ML(p,VectLinit,L1neg) #incident line from negative intersection point
    xrn,yrn,zrn = ML(pointintneg,VectL2neg,L1neg) #reflective line from negative intersection point
    return xi,yi,zi,xr,yr,zr,xin,yin,zin,xrn,yrn,zrn,pointint,pointintneg'''

'''SetRange3dIndividual (SR3I): give a radius and an origin and if a point is within that circle and on the correct side of the x axis (pos or neg) return true. '''
def SR3I (xrange, X,Y,Z, origin, sign):
    if sign == 'pos':
        if (X-origin[0])**2 + (Z-origin[2])**2 < xrange**2 and Y > 0:
            return True
        return False
    if sign == 'neg':
        if (X-origin[0])**2 + (Z-origin[2])**2 < xrange**2 and Y < 0:
            return True
        return False
    if sign != 'pos' and sign != 'neg':
        print ('Error')
        

'''Important note: Coefficients of Ellipse refers to [a,b,c] with regards to the triangle between the ellipse, center, and focus, where c is distance from focus to origin, b is furthest distance from edge to origin at right angle, and a is hypotenuse of resulting triangle''' 

'''CreateEllipseBoundShifted (CEBS): give coefficients of ellipse, the length (on x axis) wanted, and the origin this should be centered on. Creates restricted (bound and shifted) ellipse.'''
def CEBS(coeffellipse,length, origin):
    xc=np.linspace(origin[0]-float(length)/2,origin[0]+float(length)/2,100) #centers around angle
    yc1 = np.sqrt((1-(((xc-origin[0])**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #pos side of ellipse
    yc2 = -np.sqrt((1-((xc**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #neg side of ellipse
    zc = np.linspace(0,0,100) #assumes merely in the x,y plane and uses 100 points
    return xc,yc1,yc2,zc

'''RotateStrandBoundShiftCORRECTING (RSBSC): Give angle to be rotated about the x axis, coefficients of ellipse, length to restrict ellipse, origin of shifting, and sign(pos or neg if it is on the positive or negative side of the y axis). Creates the ellipse rotated at the specific angle around the x-axis'''
def RSBSC(theta, coeffellipse,length,origin, sign):
    Rotated = []
    xc,yc1,yc2,zc = CEBS(coeffellipse,length,origin)
    if sign == 'pos':
        for i in range (0,100): 
            v = [xc[i], yc1[i],zc[i]] # number of original points using POSITIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    if sign == 'neg':
        for i in range (0,100):
            v = [xc[i], yc2[i], zc[i]] #number of original points on NEGATIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta)))#multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100): #recombining into arrays of x,y,z to be plotted
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

'''CreateZBoundShiftCorrecting (CZBSC): give number of ellipses wanted, half of the angle (theta) wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated Theta about the x-axis'''
def CZBSC(a,n, coeffellipse, length, origin, sign):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a) #range from 0 to n angles in a divisions
        x,y,z = RSBSC(theta[i], coeffellipse, length, origin, sign) #ellipse for specific angle
        x1.extend(x) #adding a new ellipse for each angle
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1 #returns all ellipses

'''setrange2d(SR2): Give radius, intersection points, and origin, only keep points within the circle '''
def SR2(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti-origin[2])**2 <= xrange**2:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

'''CreateEllipseBoundShifted(CEBS): creates an ellipse with given coefficients at origin (0,0). Returns x, positive y, negative y, and z coordinates. (z is assumed to be 0 as it is 2d)'''
def CEBS(coeffellipse,length):
    xc=np.linspace(-float(length)/2,float(length)/2,100) #centers around angle
    yc1 = np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #pos side of ellipse
    yc2 = -np.sqrt((1-(((xc)**2)/(coeffellipse[0]**2)))*coeffellipse[1]**2) #neg side of ellipse
    zc = np.linspace(0,0,100) #assumes merely in the x,y plane and uses 100 points
    return xc,yc1,yc2,zc

'''RotateStrandBoundShiftCORRECTING (RSBSC): Give angle to be rotated about the x axis, coefficients of ellipse, length to restrict ellipse, origin of shifting, and sign(pos or neg if it is on the positive or negative side of the y axis). Creates the ellipse rotated at the specific angle around the x-axis'''
def RSBSC(theta, coeffellipse,length, sign):
    Rotated = []
    xc,yc1,yc2,zc = CEBS(coeffellipse,length)
    if sign == 'pos':
        for i in range (0,100): 
            v = [xc[i], yc1[i],zc[i]] # number of original points using POSITIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta))) #multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    if sign == 'neg':
        for i in range (0,100):
            v = [xc[i], yc2[i], zc[i]] #number of original points on NEGATIVE side of ellipse
            v2 = np.array(np.dot(v,Rx(theta)))#multiplied by rotation vector 
            Rotated.append(v2[0]) #rotated vectors
    xcR1 = []
    ycR1 = []
    zcR1 = []
    for j in range (0,100): #recombining into arrays of x,y,z to be plotted
        xcR1.append(Rotated[j][0])
        ycR1.append(Rotated[j][1])
        zcR1.append(Rotated[j][2])
    return xcR1,ycR1,zcR1

'''CreateZBoundShiftCorrecting (CZBSC): give number of ellipses wanted, half of the angle (theta) wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated Theta about the x-axis'''
def CZBSC(a,n, coeffellipse, length, sign):
    x1 = []
    y1 = []
    z1 = []
    for i in range (0,a):
        theta = np.linspace(0,n,a) #range from 0 to n angles in a divisions
        x,y,z = RSBSC(theta[i], coeffellipse, length, sign) #ellipse for specific angle
        x1.extend(x) #adding a new ellipse for each angle
        y1.extend(y)
        z1.extend(z)
    return x1,y1,z1 #returns all ellipses

'''FTSCEllipsoidCorrecting(FTSEC): give  number of ellipses wanted, half of the angle covered wanted (so if you want half of the ellipsoid, choose np.pi/2), coefficients of ellipse, restriction length, shift origin, and sign (pos or neg). Returns the 3d shape of the restricted and shifted ellipsoid rotated + Theta and -Theta about the x-axis to create a symmetric ellipsoid on the pos and neg side of the z plane.'''
def FTSEC (a,n, coeffellipse, length, sign):
    X,Y,Z = CZBSC(a,n, coeffellipse, length, sign) #negative side of zplane
    X1,Y1,Z1 = CZBSC(a,-n, coeffellipse, length, sign) #negative side zplane
    if sign != 'pos' and sign != 'neg':
        print ('Error')
    return X,Y,Z, X1, Y1, Z1

'''Separate: given a list of points/vectors (i.e [[x1,y1,z1],[x2,y2,z2], ...] translates into a three arrays of x, y, and z values. (this is the format for the Transform function)'''
def sep(X): 
    x,y,z = [],[],[]
    if type(X[0]) is int or type(X[0]) is float or type(X[0]) is numpy.float64:
        x = X[0]
        y = X[1]
        z = X[2]
    else:
        for i in range (0,len(X)):
            x.append(X[i][0])
            y.append(X[i][1])
            z.append(X[i][2])
    return x,y,z

'''The reverse of sep. Translates three arrays of x,y,z values back into series of [x,y,z] points/vectors. '''
def sepop(x,y,z): 
    v = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
    else:
        for i in range (0,len(x)):
            a = [x[i],y[i],z[i]]
            v.append(a)
    return v


''' rotate (V, thetaxyz) rotates a vector about a given angle in order of (x,y,z)''' 
def rotate(point,thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x,y,z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    V2 = np.array(np.dot(V,Rxyz(thetaxyz)))
    v2f = V2[0]*lenvect
    return v2f

'''rotate (V, thetaxyz) rotates a vector about a given angle in order of (z,y,x)'''
def rotaterev(point,thetaxyz):
    x = point[0]
    y = point[1]
    z = point[2]
    v = [x,y,z]
    lenvect = (x**2 + y**2 + z**2)**.5
    V = N(v)
    VZ = np.array(np.dot(V,Rz(thetaxyz[2])))
    VZY = np.array(np.dot(VZ,Ry(thetaxyz[1])))
    VZYX = np.array(np.dot(VZY,Rx(thetaxyz[0])))
    v2f = VZYX[0]*lenvect
    return v2f

'''given a point (or vector) and an origin (a local one in global coordinates), shifts to the Local origin in Global coordinates'''
def shift(point, origin):
    x = point[0]
    y = point[1]
    z = point[2]
    x2 = x + origin[0]
    y2 = y + origin[1]
    z2 = z + origin[2]
    v2 = [x2,y2,z2]
    return v2

'''Given a point (or three arrays of x,y,z for points), the GLOBAL coordinates are transformed to LOCAL coordinates where the LOCAL coordinates are defined in terms of the GLOBAL coordinate system through its GLOBAL origin and GLOBAL rotation. Essentially transforms point(s) from global coordinate system to given local coordinate system. the origin is the LOCAL origin in GLOBAL coordinates'''
def transformGL(x,y,z,origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
        if x ==0 and y ==0 and z == 0:
            vf = shift(v,negvect(origin))
            XTR=vf[0]
            YTR=vf[1]
            ZTR=vf[2]
        else:
            v2S = shift(v,negvect(origin))
            v2RS = rotaterev(v2S,negvect(thetaxyz))
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range (0, len(x)):
            v = [x[i],y[i],z[i]]
            if x[i] == 0 and y[i] ==0 and z[i] == 0:
                vf = shift(v,negvect(origin))
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2S = shift(v,negvect(origin))
                v2RS = rotaterev(v2S,negvect(thetaxyz))
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])    
    return XTR,YTR,ZTR

'''transforms point(s) from local coordinate system to corresponding glocal coordinate system. origin is the LOCAL origin in GLOBAL coordinates'''
def transformLG(x,y,z,origin, thetaxyz):
    XTR = []
    YTR = []
    ZTR = []
    if type(x) is int or type(x) is float or type(x) is numpy.float64:
        v = [x,y,z]
        if x ==0 and y ==0 and z == 0:
            vf = shift(v,origin)
            XTR=vf[0]
            YTR=vf[1]
            ZTR=vf[2]
        else:
            v2R = rotate(v,thetaxyz)
            v2RS = shift(v2R,origin)
            XTR = v2RS[0]
            YTR = v2RS[1]
            ZTR = v2RS[2]
    else:
        for i in range (0, len(x)):
            v = [x[i],y[i],z[i]]
            if x[i] == 0 and y[i] ==0 and z[i] == 0:
                vf = shift(v,origin)
                XTR.append(vf[0])
                YTR.append(vf[1])
                ZTR.append(vf[2])
            else:
                v2R = rotate(v,thetaxyz)
                v2RS = shift(v2R,origin)
                XTR.append(v2RS[0])
                YTR.append(v2RS[1])
                ZTR.append(v2RS[2])    
    return XTR,YTR,ZTR

'''EllipseLineIntShiftCorrecting(ELISC): give point on line, vector of line, coefficients of ellipse (assuming rotated around x axis), and shifted origin (assuming origin is still at z = 0). Returns intersection points. 
def ELISC(pli1,v1,coeffellipse): #given point of line, vector, and the axes of the ellipse, find the intersection
    A = (1/coeffellipse[0]**2) + (v1[1]**2)/((v1[0]**2)*(coeffellipse[1])**2) + (v1[2]**2)/((v1[0]**2)*(coeffellipse[0]**2))
    B1 = -(2*pli1[0]*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) -(2*pli1[0]*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    B2 = (2*v1[1]*pli1[1])/((v1[0])*coeffellipse[1]**2)+ (2*v1[2]*pli1[2])/((v1[0])*coeffellipse[0]**2)
    B = B1 + B2
    C1 = ((pli1[0]**2)*(v1[1]**2))/((v1[0]**2)*coeffellipse[1]**2) + ((pli1[0]**2)*(v1[2]**2))/((v1[0]**2)*coeffellipse[0]**2)
    C2 = -(2*pli1[0]*v1[1]*pli1[1])/((v1[0])*(coeffellipse[1]**2)) -(2*pli1[0]*v1[2]*pli1[2])/((v1[0])*(coeffellipse[0]**2))
    C3 = (pli1[2]**2)/(coeffellipse[0]**2) + (pli1[1]**2)/(coeffellipse[1]**2)
    C = C1 + C2 + C3 - 1
    xint = [(-B + np.sqrt(B**2 - 4*A*C))/(2*A),(-B - np.sqrt(B**2 - 4*A*C))/(2*A)]
    yint = [((xint[0] - pli1[0])/(v1[0]))*v1[1] + pli1[1],((xint[1] - pli1[0])/(v1[0]))*v1[1] + pli1[1]]
    zint = [((xint[0] - pli1[0])/(v1[0]))*v1[2] + pli1[2], ((xint[1] - pli1[0])/(v1[0]))*v1[2] + pli1[2]]
    #xintshift = [x + origin[0] for x in xint]
    #yintshift = [y + origin[1] for y in yint]
    #yintshift = yint #does y not change what?
    #zintshift = [z + origin[2] for z in zint] (typically centered at zero)
    return xint,yint,zint #returns in frame of ellipse'''

'''ReflectEllipsePointCorrecting(REPC): give ellipse coefficients, point on line, vector of line, ellipse origin, and sign. Returns incident rays, reflecting rays, and intersection points for positive and negative intersection. Uses sign to determine correct direction of reflecting rays.   
def REPC(coeffellipse,pli,v, sign):
    c = TC(pli,v,coeffellipse) #plane coefficients
    cpos = c[0]
    cneg = c[1]
    VectL = v
    VectLNorm = N(v) #incident unit vector
    dU, Npos = PI(cpos) #gradient and normal of plane of positive int
    dUneg, Nneg = PI(cneg)
    VectL2 = VectLNorm - 2*Npos
    VectL2neg = VectLNorm-2*Nneg
    #check if it is going in the right direction for positive int
    if sign == 'neg' and VectL2[1] < 0 : 
        VectL2 = [-x for x in VectL2]
    if sign == 'pos' and VectL2[1] > 0:
        VectL2 = [-x for x in VectL2]
    #check if it is going in the right direction for negative int
    if sign == 'neg' and VectL2neg[1] < 0 : 
        VectL2neg = [-x for x in VectL2neg]
    if sign == 'pos' and VectL2neg[1] > 0:
        VectL2neg = [-x for x in VectL2neg]
    #VectLNorm2 = norm(VectL2) #reflected unit vector
    #xp,yp,zp = MP(c,r) #plane
    #xint,yint,zint = ELISC(pli,v,coeffellipse) #in frame of ellipse
    xint,yint,zint = ELI(pli,v,coeffellipse)
    #pointintRETURN = [xint,yint,zint]
    pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    pointintneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    L1 = np.sqrt((pointint[0] - pli[0])**2 + (pointint[1] - pli[1])**2 + (pointint[2] - pli[2])**2)
    L1neg = np.sqrt((pointintneg[0] - pli[0])**2 + (pointintneg[1] - pli[1])**2 + (pointintneg[2] - pli[2])**2)
    xi,yi,zi = ML(pli,VectL,L1) #incident line from intersection point
    xr,yr,zr = ML(pointint,VectL2,L1)
    xin,yin,zin = ML(pli,VectL,L1neg) #incident line from intersection point
    xrn,yrn,zrn = ML(pointintneg,VectL2neg,L1neg)
    return xi,yi,zi,xr,yr,zr,xin,yin,zin,xrn,yrn,zrn, pointint, pointintneg, VectL2, VectL2neg''' 

''' SetRange3dIndCorrecting (SR3IC): simply changing requirements. same as SR3I, just ignoring z.'''
def SR3IC (xrange, X,Y,Z, ellipseorigin, sign): #given range, one point, origin, if it lies in or not
    if sign == 'pos':
        if (X-ellipseorigin[0])**2 + (Z-ellipseorigin[2])**2 < xrange**2 and (Y-ellipseorigin[1]) > 0:
            return True
        return False
    if sign == 'neg':
        if (X-ellipseorigin[0])**2 + (Z-ellipseorigin[2])**2 < xrange**2 and (Y-ellipseorigin[1]) < 0:
            return True
        return False
    if sign != 'pos' and sign != 'neg':
        print ('Error')
        
''' SetRange3Both (SR3B): simply changing requirements. same as SR3IC but only if a point is within an ellipse.'''
def SR3B (ranges, xinti,yinti,zinti, origin): #given range, one point, origin, if it lies in or not
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    if ( (((xinti-xc)**2/xr**2) + ((yinti-yc)**2/yr**2)) <= 1 
        and (((yinti-yc)**2/yr**2) + ((zinti-zc)**2/zr**2)) <=1
        and (((zinti-zc)**2/zr**2) + ((xinti-xc)**2/xr**2)) <=1):
        return True
    return False

'''reflect_specellipsePOINTCORRECTING (RSEPC): Give coeffients of ellipse, points of rays, vectors of rays, radius of target, origin of target, and sign. Returns initial rays, reflecting rays, intersection points, and reflecting vectors. 
def RSEPC(coeffellipse,pli,vectors, xrange, ellipseorigin,ellipsethetaxyz, sign):
    xi = []
    yi = []
    zi = []
    xr = []
    yr = []
    zr = []
    Vect = []
    pointints = []
    if type(pli[0]) is int or type(pli[0]) is float: #assuming it is the source
        for i in range (0,len(vectors)):
            Vi = vectors[i]
            Vi2 = np.array(np.dot(Vi,Rx(np.pi/2)))
            xiL,yiL,ziL,xrL,yrL,zrL,xinL,yinL,zinL,xrnL,yrnL,zrnL, pointint, pointintneg = REP(coeffellipse,Vi2[0],pli)
            if SR3IC(xrange, pointint[0], pointint[1], pointint[2], ellipseorigin, sign) == True: 
                xi.append(xiL)
                yi.append(yiL)
                zi.append(ziL)
                xr.append(xrL)
                yr.append(yrL)
                zr.append(zrL)
                pointints.append(pointint)
                Vect.append(Vi2[0])
            if SR3IC(xrange, pointintneg[0], pointintneg[1], pointintneg[2], ellipseorigin, sign) == True:
                xi.append(xinL)
                yi.append(yinL)
                zi.append(zinL)
                xr.append(xrnL)
                yr.append(yrnL)
                zr.append(zrnL)
                pointints.append(pointintneg)
                Vect.append(Vi2[0])
    else:
        for i in range (0, len(vectors)):
            Vi = vectors[i]
            Pli = pli[i] #(or pli/original points of lines)
            xiL,yiL,ziL,xrL,yrL,zrL,xinL,yinL,zinL,xrnL,yrnL,zrnL, pointint, pointintneg, vectL2, vectL2neg = REPC(coeffellipse,Pli,Vi, sign)
            if SR3IC(xrange, pointint[0], pointint[1], pointint[2],ellipseorigin, sign) == True: 
                xi.append(xiL)
                yi.append(yiL)
                zi.append(ziL)
                xr.append(xrL)
                yr.append(yrL)
                zr.append(zrL)
                pointints.append(pointint) 
                Vect.append(vectL2)
            if SR3IC(xrange, pointintneg[0], pointintneg[1], pointintneg[2], ellipseorigin, sign) == True:
                xi.append(xinL)
                yi.append(yinL)
                zi.append(zinL)
                xr.append(xrnL)
                yr.append(yrnL)
                zr.append(zrnL)
                pointints.append(pointintneg)
                Vect.append(vectL2neg)
    return xi,yi,zi,xr,yr,zr, pointints, Vect'''

'''Negates a vector (i.e. from [x,y,z] to [-x,-y,-z]) '''
def negvect(vect):
    if type(vect[0]) is int or type(vect[0]) is float or type(vect[0]) is numpy.float64:
        vectset = [-x for x in vect]
    else:
        vectset = [[-y for y in x] for x in vect]
    return vectset

'''Spec: give the number of rays wanted, returns specular distribution of n vectors. Adapted from Meyer's Specular notebook. '''
def spec(n):
    x,y,z = [],[],[]
    for i in np.arange(n):
        theta=np.arccos(uniform(-1,1))
        phi=np.random.uniform(0,2*np.pi)
        xt=np.sin(theta)*np.cos(phi)
        yt=np.sin(theta)*np.sin(phi)
        zt=np.cos(theta)
        if zt<0.:
            zt=-zt
        a=uniform(0,1)
        while a>zt:
            theta=np.arccos(uniform(-1,1))
            phi=np.random.uniform(0,2*np.pi)
            xt=np.sin(theta)*np.cos(phi)
            yt=np.sin(theta)*np.sin(phi)
            zt=np.cos(theta)
            if zt<0.:
                zt=-zt
            a=uniform(0,1)
        x=np.append(x,xt)
        y=np.append(y,yt)
        z=np.append(z,zt)
    V = []
    for i in np.arange(n):
        v = [x[i], y[i], z[i]]
        V.append(v)  
    return V

#now, given a point and the vector, rewrite reflection to take that, and give out that.
# rewrite functions
'''ReflectEllipsePointCorrectingNEW(REPCN): give ellipse coefficients, point on line, vector of line, and sign. Returns points of intersection and vectors of reflection. Uses sign to determine correct direction of reflecting rays.   ''' 
def REPCN(coeffellipse,pli,v, sign):
    c = TC(pli,v,coeffellipse) #plane coefficients
    cpos = c[0]
    cneg = c[1]
    VectL = v
    VectLNorm = N(v) #incident unit vector
    dU, Npos = PI(cpos) #gradient and normal of plane of positive int
    dUneg, Nneg = PI(cneg)
    VectL2 = VectLNorm - 2*Npos
    VectL2neg = VectLNorm-2*Nneg
    #check if it is going in the right direction for positive int
    if sign == 'neg' and VectL2[1] < 0 : 
        VectL2 = [-x for x in VectL2]
    if sign == 'pos' and VectL2[1] > 0:
        VectL2 = [-x for x in VectL2]
    #check if it is going in the right direction for negative int
    if sign == 'neg' and VectL2neg[1] < 0 : 
        VectL2neg = [-x for x in VectL2neg]
    if sign == 'pos' and VectL2neg[1] > 0:
        VectL2neg = [-x for x in VectL2neg]
    xint,yint,zint = ELI(pli,v,coeffellipse)
    pointint = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    pointintneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    #L1 = np.sqrt((pointint[0] - pli[0])**2 + (pointint[1] - pli[1])**2 + (pointint[2] - pli[2])**2)
    #L1neg = np.sqrt((pointintneg[0] - pli[0])**2 + (pointintneg[1] - pli[1])**2 + (pointintneg[2] - pli[2])**2)
    return pointint, pointintneg, VectL2, VectL2neg


'''Reflect_SpecellipsePointCorrectingNew (RSEPCN): Give coeffients of ellipse, points of rays, vectors of rays, radius of target, local origin, local rotation (typically [0,0,0] and [0,0,0]. So given source as points and vectors, returns intersection points and vectors in TERMS OF THE LOCAL ORIGIN. '''
def RSEPCN(coeffellipse,pli,vectors, xrange, ellipseorigin,ellipsethetaxyz, sign):
    Vect = []
    pointints = []
    if sign == 'both': #both positive and negative side of ellipse
        print ('oops')
    else:
        if len(pli) == 0:
            return [],[]
        if type(pli[0]) is int or type(pli[0]) is float: #assuming it is a source from one point
            for i in range (0,len(vectors)):
                pointint, pointintneg,VectL2, VectL2neg = REPCN(coeffellipse,pli,vectors[i], sign)
                if SR3IC(xrange, pointint[0], pointint[1], pointint[2], ellipseorigin, sign) == True: 
                    pointints.append(pointint)
                    Vect.append(VectL2)
                if SR3IC(xrange, pointintneg[0], pointintneg[1], pointintneg[2], ellipseorigin, sign) == True:
                    pointints.append(pointintneg)
                    Vect.append(VectL2neg)
        else:
            for i in range (0, len(vectors)):
                Vi = vectors[i]
                Pli = pli[i] #(or pli/original points of lines)
                pointint, pointintneg, vectL2, vectL2neg = REPCN(coeffellipse,Pli,Vi, sign)
                if SR3IC(xrange, pointint[0], pointint[1], pointint[2],ellipseorigin, sign) == True: 
                    pointints.append(pointint) 
                    Vect.append(vectL2)
                if SR3IC(xrange, pointintneg[0], pointintneg[1], pointintneg[2], ellipseorigin, sign) == True:
                    pointints.append(pointintneg)
                    Vect.append(vectL2neg)
    return pointints, Vect

'''ReTransform (RT): given points and vectors, transforms from one coordinate system (given by thet and origin with respect to GLOBAL coordinate system) to another coordinate system (given by thet and origin with respect to GLOBAL coordinate system) '''
def RT(sourcepoints,v1,sourcethet1, ellipseorigin1, sourcethet2, ellipseorigin2):
    if len(sourcepoints) == 0:
        return [],[]
    spx,spy,spz = sep(sourcepoints)
    vx,vy,vz = sep(v1)
    vectorigin = [0,0,0] #don't shift vectors
    #LOCAL to GLOBAL
    vGx,vGy,vGz = transformLG(vx,vy,vz,vectorigin,sourcethet1)
    spGx,spGy,spGz = transformLG(spx,spy,spz,ellipseorigin1,sourcethet1)
    #GLOBAL back to SECOND LOCAL
    vfx,vfy,vfz = transformGL(vGx,vGy,vGz,vectorigin,sourcethet2)
    spfx,spfy,spfz = transformGL(spGx,spGy,spGz,ellipseorigin2,sourcethet2)
    sp = sepop(spfx,spfy,spfz)
    v2 = sepop(vfx,vfy,vfz)
    return sp, v2

'''Select Range specifically for ellipse 7 (see page 131 sheets) and 143.'''
def SR10(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti <0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

def SR7(xrange,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if (xinti-origin[0])**2 + (zinti)**2 <= xrange**2 and yinti >0:
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

''' Select Range specifically for ellipse 10 (see page 133 sheets). specifically for taking into account extreme rotation for E10'''
def SR103d(ranges,X,Y,Z, origin):
    xintG = []
    yintG = []
    zintG = []
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if ( (((xinti-xc)**2/xr**2) + ((yinti-yc)**2/yr**2)) <= 1 
            and (((yinti-yc)**2/yr**2) + ((zinti-zc)**2/zr**2)) <=1
            and (((zinti-zc)**2/zr**2) + ((xinti-xc)**2/xr**2)) <=1):
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

'''Creates a circular source with a given radius'''
def circularsource(r): #radius
    xpoint = []
    ypoint = []
    zpoint = []
    for x in range(-r,r):
        for y in range(-r,r):
            d = numpy.sqrt((x**2) + (y**2))
            if d <= r: 
                xpoint.append(x)
                ypoint.append(y)
                zpoint.append(int(0))
    return xpoint, ypoint, zpoint


#given sourcepoint = [x,y,z] where x etc is an array
'''FormSource: give number of rays, potential source points, the GLOBAL angle, and GLOBAL origin (that the source should be made with respect to). Returns random collection of points and vectors. '''
def FS(specnum,sourcepoint,sourcethet,origin):
    originG = [0,0,0]
    if type(sourcepoint[0]) is int or type(sourcepoint[0]) is float or type(sourcepoint[0]) is numpy.float64:
        v1 = spec(specnum)
        vx,vy,vz = sep(v1)
        v1x,v1y,v1z = transformLG(vx,vy,vz,originG,sourcethet)
        p1x,p1y,p1z = shift(sourcepoint,origin)
        sp = [p1x,p1y,p1z]
        v2 = sepop(v1x,v1y,v1z)
    else: 
        v1 = spec(specnum)
        vx,vy,vz = sep(v1)
        v1x,v1y,v1z = transformLG(vx,vy,vz,originG,sourcethet)
        v2 = sepop(v1x,v1y,v1z)
        sp = []
        for i in range (0,specnum):
            spx = random.choice(sourcepoint[0])
            spy = random.choice(sourcepoint[1])
            spz = random.choice(sourcepoint[2])
            #sp1x,sp1y,sp1z = transformLG(spx,spy,spz,origin,sourcethet)
            spT = [spx,spy,spz]
            sp.append(spT)
    return sp,v2

'''creates a list of potential source points within a certain range, tilted at a certain angle, corresponding to a specific origin.'''
def specsource(r,origin,thet):
    x,y,z=circularsource(r)
    x1,y1,z1 = transformLG(x,y,z,origin,thet)
    sourcepoint = [x1,y1,z1]
    return sourcepoint

'''from the xrange being determined in GLOBAL coordinate system, translates from Global to Local. Returns center point and the xrange'''
def xrangeGL6 (x1,y1,z1,x3,y3,z3,origin,thet):
    x,y,z = [],[],[]
    x2,y2,z2 = transformGL(x1,y1,z1,origin,thet)
    x4,y4,z4 = transformGL(x3,y3,z3,origin,thet)
    x.extend(x2),x.extend(x4),y.extend(y2),y.extend(y4),z.extend(z2),z.extend(z4)
    
    xrangeL = np.sqrt((min(x) - max(x))**2)/2
    yrangeL = np.sqrt((min(y) - max(y))**2)/2
    zrangeL = np.sqrt((min(z) - max(z))**2)/2
    
    xcenter = min(x) + xrangeL
    ycenter = min(y) + yrangeL
    zcenter = min(z) + zrangeL
    
    xcenterL = [xcenter,ycenter,zcenter]
    xrangesL = [xrangeL,yrangeL,zrangeL]
    return xcenterL,xrangesL

def SR103di(ranges,X,Y,Z, origin): #this corrects SR103d
    xintG = []
    yintG = []
    zintG = []
    xr = ranges[0]
    yr = ranges[1]
    zr = ranges[2]
    xc = origin[0]
    yc = origin[1]
    zc = origin[2]
    for i in range (0,len(X)):
        xinti = X[i]
        yinti = Y[i]
        zinti = Z[i]
        if ( ((xinti-xc)**2)/(xr**2) + ((yinti-yc)**2)/(yr**2) <= 1 
            and ((yinti-yc)**2)/(yr**2) + ((zinti-zc)**2)/(zr**2) <=1
            and ((zinti-zc)**2)/(zr**2) + ((xinti-xc)**2)/(xr**2) <=1):
            xintG.append(xinti)
            yintG.append(yinti)
            zintG.append(zinti)
    return xintG,yintG,zintG

#given original point and vector from it, figure out properties of intersection wanted
#returns x2 greater (G) than or less (L) than x1
def finddirec(p1,v,intpos,intneg,vectpos,vectneg):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    xpos = intpos[0]
    ypos = intpos[1]
    zpos = intpos[2]
    xneg = intneg[0]
    yneg = intneg[1]
    zneg = intneg[2]
    direc = [] 
    direcpos = []
    direcneg = []
    if v1 >=0:
        direc.append('G')
    if v1 < 0:
        direc.append('L')
    if v2 >=0:
        direc.append('G')
    if v2 < 0:
        direc.append('L')
    if v3 >=0:
        direc.append('G')
    if v3 < 0:
        direc.append('L')
    if xpos>=x1:
        direcpos.append('G')
    if xpos < x1:
        direcpos.append('L')
    if ypos>=y1:
        direcpos.append('G')
    if ypos < y1:
        direcpos.append('L')
    if zpos>=z1:
        direcpos.append('G')
    if zpos < z1:
        direcpos.append('L')
    if xneg>=x1:
        direcneg.append('G')
    if xneg < x1:
        direcneg.append('L')
    if yneg>=y1:
        direcneg.append('G')
    if yneg < y1:
        direcneg.append('L')
    if zneg>=z1:
        direcneg.append('G')
    if zneg < z1:
        direcneg.append('L')
    if direc == direcpos:
        return intpos, vectpos
    else:
        if direc == direcneg:
            return intneg, vectneg
        else: 
            return 'Error'

    
def REPCNi(coeffellipse,pli,v):
    Npos,Nneg = NormalP(pli,v,coeffellipse) #plane coefficients
    VectLNorm = N(v) #incident unit vector
    Npos = np.array([-x for x in Npos]) 
    Nneg = np.array([-x for x in Nneg])
    vectpos = VectLNorm - 2*N(Npos)*(np.dot(VectLNorm,N(Npos)))
    vectneg = VectLNorm - 2*N(Nneg)*(np.dot(VectLNorm,N(Nneg)))
    xint,yint,zint = ELI(pli,v,coeffellipse)
    intpos = [float(xint[0]),float(yint[0]),float(zint[0])] #array and points of intersection
    intneg = [float(xint[1]),float(yint[1]),float(zint[1])] #array and points of intersection
    GoodInt,GoodVect = finddirec(pli,v,intpos,intneg,vectpos,vectneg)
    return GoodInt, GoodVect

#uses 3d ellipses for range, ellipse origin for center of DESIRED range, 
def RSEPCNi(coeffellipse,pli,vectors, ranges, ellipseorigin):
    Vect = []
    pointints = []
    if len(pli) == 0:
        return [],[]
    if type(pli[0]) is int or type(pli[0]) is float: #assuming it is a source from one point
        for i in range (0,len(vectors)):
            Gpoint,Gvect = REPCNi(coeffellipse,pli,vectors[i])
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    else:
        for i in range (0, len(pli)):
            Vi = vectors[i]
            Pli = pli[i] #(or pli/original points of lines)
            Gpoint,Gvect = REPCNi(coeffellipse,Pli,Vi)
            if SR3B(ranges, Gpoint[0],Gpoint[1],Gpoint[2], ellipseorigin) == True:
                pointints.append(Gpoint)
                Vect.append(Gvect)
    return pointints, Vect


