from RayTraceFun_forInt import *
from Background import *
def TTTT(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_TP1,Ray_E8,Ray_TP2,Ray_E3,Ray_M0,Ray_E4,Ray_TP3,Ray_E5,Ray_TP4
    #return Ray_TP4
def RRRR(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_RP4
def TTTR(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_RP4
def TTRT(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_TP4
def TRTT(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_TP4
def RTTT(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_TP4
def TTRR(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_RP4
def TRTR(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_RP4
def RTRT(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_TP4
def TRRT(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_TP4
def RRTT(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_TP4
def RTTR(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_RP4
def TRRR(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_RP4
def RTRR(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_RP4
def RRTR(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E5,coeffpolar,originpolar4,p4)
    return Ray_RP4
def RRRT(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E6,coeffpolar,originpolar4,p4)
    return Ray_TP4

def detectypos(Ri,p1,p2,p3,p4):
    R1 = TTTR(Ri,p1,p2,p3,p4)
    R2 = TTRT(Ri,p1,p2,p3,p4)
    R3 = TRTT(Ri,p1,p2,p3,p4)
    R4 = RTTT(Ri,p1,p2,p3,p4)
    R5 = TRRR(Ri,p1,p2,p3,p4)
    R6 = RTRR(Ri,p1,p2,p3,p4)
    R7 = RRTR(Ri,p1,p2,p3,p4)
    R8 = RRRT(Ri,p1,p2,p3,p4)
    return(R1,R2,R3,R4,R5,R6,R7,R8)

def detectyneg(Ri,p1,p2,p3,p4):
    R1 = TTTT(Ri,p1,p2,p3,p4)
    R2 = RRRR(Ri,p1,p2,p3,p4)
    R3 = TTRR(Ri,p1,p2,p3,p4)
    R4 = RTTR(Ri,p1,p2,p3,p4)
    R5 = RTRT(Ri,p1,p2,p3,p4)
    R6 = TRRT(Ri,p1,p2,p3,p4)
    R7 = RRTT(Ri,p1,p2,p3,p4)
    R8 = TRTR(Ri,p1,p2,p3,p4)
    Ray_E71 = ReflEll(R1,thet7,origin7,coeffellipse7)
    Ray_E72 = ReflEll(R2,thet7,origin7,coeffellipse7)
    Ray_E73 = ReflEll(R3,thet7,origin7,coeffellipse7)
    Ray_E74 = ReflEll(R4,thet7,origin7,coeffellipse7)
    Ray_E75 = ReflEll(R5,thet7,origin7,coeffellipse7)
    Ray_E76 = ReflEll(R6,thet7,origin7,coeffellipse7)
    Ray_E77 = ReflEll(R7,thet7,origin7,coeffellipse7)
    Ray_E78 = ReflEll(R8,thet7,origin7,coeffellipse7)
    return Ray_E71,Ray_E72,Ray_E73,Ray_E74,Ray_E75,Ray_E76,Ray_E77,Ray_E78

'''all the below are the entire fts run through given an initial ray and four polarizers '''

def TTTTE(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def RRRRE(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def TTRRE(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_TP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_RP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)    
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def RTTRE(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def RTRTE(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_TP2 = IntPolT2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_TP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def TRRTE(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_RP3 = IntPolR2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_RP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def RRTTE(Ri,p1,p2,p3,p4):
    Ray_RP1 = IntPolR2(Ri,coeffpolar,originpolar1,p1)#p1
    Ray_E9 = ReflEll(Ray_RP1,thet5,origin9,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E9,coeffpolar,originpolar2,p2) #P2
    Ray_E3 = ReflEll(Ray_RP2,thet,origin3,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E3, coeffmirr, originG) #off mirror
    Ray_E4 = ReflEll(Ray_M0, thet,origin4,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E4,coeffpolar,originpolar3,p3) #P3
    Ray_E5 = ReflEll(Ray_TP3, thet5,origin5,coeffellipse56)
    Ray_TP4 = IntPolT2(Ray_E5,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_TP4,thet7,origin7,coeffellipse7)
    return Ray_E72

def TRTRE(Ri,p1,p2,p3,p4):
    Ray_TP1 = IntPolT2(Ri,coeffpolar,originpolar1,p1) #P1
    Ray_E8 = ReflEll(Ray_TP1,thet6,origin8,coeffellipse56) #E8
    Ray_RP2 = IntPolR2(Ray_E8,coeffpolar,originpolar2,p2) #P2
    Ray_E1 = ReflEll(Ray_RP2,thet,origin1,coeffellipse) #E3
    Ray_M0 = IntM2(Ray_E1, coeffmirr, originG) #off mirror
    Ray_E2 = ReflEll(Ray_M0, thet,origin2,coeffellipse) #off E4
    Ray_TP3 = IntPolT2(Ray_E2,coeffpolar,originpolar3,p3) #P3
    Ray_E6 = ReflEll(Ray_TP3, thet6,origin6,coeffellipse56)
    Ray_RP4 = IntPolR2(Ray_E6,coeffpolar,originpolar4,p4)
    Ray_E72 = ReflEll(Ray_RP4,thet7,origin7,coeffellipse7)
    return Ray_E72