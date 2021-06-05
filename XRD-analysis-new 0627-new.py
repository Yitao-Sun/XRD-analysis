import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import math as m
import re
import os
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
import datetime
import time

###   plotXRD   ###
def PlotXRD(filename,x,y):
    pp=[0]*5
    f1=open(filename,'r')    
    rl=f1.readlines()
    maxn=int(len(rl))
    agl=[0]*maxn    #angel
    count=[0]*maxn    #count

    for j in range(maxn):
        pp=rl[j].split(',')
        agl[j]=float(pp[0])
        count[j]=float(pp[1])
        
    ###   delete Si peaks   ###
    agl=agl[:980]+agl[1040:3150]+agl[3220:]
    count=count[:980]+count[1040:3150]+count[3220:]
    ################################

    plt.subplot(xlength, ylength, (ylength-y-1)*xlength+x+1)
    plt.plot(agl,count,linewidth=3,color='r')
    plt.title('X='+str(x)+' Y='+str(y),fontsize=32)
    plt.tick_params(labelsize=24)

    return
################################
###   Background Subtract 2  ###
def background_subtract2(q,count):
    Q, C = set_window(30, q, count)  ###number of points to use
##    print(Q,C)
    Q, C = get_minimal(Q,C)
##    print(Q,C)
    f = interp1d(Q, C, kind='cubic', bounds_error=False,\
                 fill_value='extrapolate')
    
    bq=[0]*len(q)
    bcount=[0]*len(q)
    for i in range(len(bq)):
        bq[i]=q[i]
        bcount[i]=count[i]-f(q[i])
    return bq,bcount

def get_minimal(Q,C):
    serial=range(len(Q))
    for i in serial[1:-1]:
        temp=[C[i]]
        for i1 in serial[:i]:
            for i2 in serial[i+1:]:
                temp.append((Q[i]-Q[i1])/(Q[i2]-Q[i1])*(C[i2]-C[i1])+C[i1])        
        C[i]=min(temp)
    return Q,C

def func(x, *params):
    params = params[0]
##    y = chebval(x, params[:4])
##    E = params[4]
##    y = y + E /x
    y = params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]/x
    return y


def set_window(N, Q_original, intensity_original):
    Q = []
    intensity = []
    for i in np.linspace(9,len(Q_original)-10,N):
        Q.append(np.mean(Q_original[int(i-9):int(i+9)]))
        intensity.append(np.mean(intensity_original[int(i-9):int(i+9)]))
    Q = np.array(Q)
    intensity = np.array(intensity)
    return Q, intensity
##########   end   #############
################################

######################
###   find peaks   ###
def FindPeaks(q,count):
    tw=int(50)    #test width
    threshold=0.05*np.max(count)
    threshold=0.075
    threshold=np.std(count[0:50])*6
    peaks=[]
    for j in range(tw,len(q)-tw):
        if (count[j]==np.max(count[j-int(tw/2):j+int(tw/2)])):
            ave1=np.mean(count[j-2:j+3])
            ave2=np.mean(count[j-tw-2:j-tw+3])
            ave3=np.mean(count[j+tw-2:j+tw+3])
            #print(j,agl[j],ave1,ave2,ave3)
            if (ave1-ave2>threshold and ave1-ave3>threshold):
                peaks.append([q[j],count[j]])
    return peaks
##########   end   #############
################################

###########################
###   fit to get FWHW   ###
def GetFWHM(q,count,pieces):   
    a0=np.max(count[int(pieces/6):int(pieces/2)])
    b0=q[count.index(a0)]    
    Q=[]
    C=[]
    for i in range(len(q)):
        if (q[i]>b0-1 and q[i]<b0+1):
            Q.append(q[i])
            C.append(count[i])

    p0=[0.5,b0,0.5,a0]    
    bounds=([0,0,0,-np.inf],[np.inf,np.inf,1,np.inf])
    popt, pcov = curve_fit(funcGLS, Q, C, p0=p0,bounds=bounds)

    Qerr=[]
    Cerr=[]
    for i in range(len(Q)):
        if (abs(Q[i]-b0)<1):
            Cerr.append(C[i])
            Qerr.append(Q[i])
    perr = np.linalg.norm((2.5-1.5*abs(Qerr-b0))*(Cerr-funcGLS(Qerr, *popt)))
      
    return popt,perr

def funcGLS(x,F,E,M,h):
##    F,E,M,h=p
    result = h*(1-M)*m.e**(-4*m.log(2)*(x-E)**2/F**2)\
             +h*M/(1+4*(x-E)**2/F**2)
    return result

def funcGaus(x,p):
    a,b,c,d=p
    return a*m.e**(-(x-b)**2/2/c**2)+d
##########   end   #############
################################

###########################
###   dataproc   ###
def dataproc(bq,bcount,pieces):
    tempq=[[] for i in range(pieces)]
    tempcount=[[] for i in range(pieces)]
    for i in range(len(bq)):        
        temp=pieces*(bq[i]-bq[0])/(bq[-1]-bq[0]+0.01)
        tempq[int(temp)].append(bq[i])
        tempcount[int(temp)].append(bcount[i])
    for i in range(pieces):
        tempq[i]=np.mean(tempq[i])
        tempcount[i]=np.mean(tempcount[i])
    for i in range(pieces):
        if (m.isnan(tempq[i])):
            tempq[i]=(tempq[i-1]+tempq[i+1])/2
            tempcount[i]=(tempcount[i-1]+tempcount[i+1])/2
    top=np.max(tempcount)
    for i in range(pieces):
        tempcount[i]=tempcount[i]/top
    for i in range(len(bq)):
        bcount[i]=bcount[i]/top
    return tempq,tempcount,bq,bcount,top        
##########   end   #############
################################

def GetInfo(filename,lamda,x,y):
    
    ax=plt.subplot(xlength, ylength, (y-1)*ylength+(xlength-x))
    plt.title('X='+str(-int(5*(x-10)))+' Y='+str(-int(5*(y-10))),fontsize=32)
    plt.xticks(fontsize=24)
##    plt.yticks([],[])
    
    pp=[0]*5
    F,E,M,h,perr,F_a,E_a,cat=0,0,0,0,0,0,0,0
    f1=open(filename,'r')    
    rl=f1.readlines()
    maxn=int(len(rl))
    agl=[0]*maxn    #angel
    count=[0]*maxn    #count
    pieces=int(100)
    
    for j in range(maxn):
        pp=rl[j].split(',')
        agl[j]=float(pp[0])
        count[j]=float(pp[1])

    ###   angle to q   ###
    q=[]
    for theta in agl:
        q.append(4*np.pi*np.sin(theta*np.pi/360)/lamda)

        ### avoid Si peaks ###
    ######################
    q=q[:980]+q[1040:3150]+q[3220:]
    count=count[:980]+count[1040:3150]+count[3220:]
    ######################

##    print(np.max(count),q,count)

    ###   background subtract   ###
    bq,bcount=background_subtract2(q,count)
    proq,procount,bq,bcount,top=dataproc(bq,bcount,pieces)
    for i in range(len(count)):
        count[i]=count[i]/top
    
    ###   plot original data   ###    
    plt.plot(q,count,'-.',markersize=1,linewidth=1,color='k',alpha=0.5)
    ###   plot background subtracted data   ###    
    plt.plot(bq,bcount,'.',markersize=3,color='b')
    ###   plot nomalized data   ###
    plt.plot(proq,procount,'.',markersize=10,color='r')

    ###   find peaks   ###
    y_axis=count
    x_axis=q
    peaks = FindPeaks(q,count)
    for peak in peaks:
        plt.plot([peak[0],peak[0]], [0,np.max(count)*0.3],linewidth=3, color="red")

    if (len(peaks)>0):
        cat=1
    else:
        popt,perr=GetFWHM(proq,procount,pieces)
        F,E,M,h=popt
        cat=2

    if (F>0):
        x=np.linspace(1.3,4.7,1000)
        y=[]
        for i in x:
            y.append(funcGLS(i,F,E,M,h))
        plt.plot(x,y,linewidth=5,color='g')
        text='%.4f' %F
        plt.text(0.65,0.8,text,fontsize=24,transform=ax.transAxes)
        print(perr)
        text='%.4f' %perr
        plt.text(0.65,0.6,text,fontsize=24,transform=ax.transAxes)
    print('fitting done')
    print(F,E,M,h)

    F_a=m.asin((E+F/2)*lamda/4/np.pi)/np.pi*360-\
         m.asin((E-F/2)*lamda/4/np.pi)/np.pi*360
    E_a=m.asin(E*lamda/4/np.pi)/np.pi*360

    
    return cat,F,E,M,h,perr,len(peaks),F_a,E_a
    #2 for glass; 1 for crystal; 3 for mixed

'''
Main script
'''
##############################
##############################
##   main structure begin   ##

lamda=1.54   ### unit is Angstrom
xlength=20
ylength=20

###   get 'com'   ###
lst=os.listdir()
for i in lst:
    a,b=i.split('.')
    if (b=='data'):
        com=a.split('_')[0]
        break

start = datetime.datetime.now()
##############################
###   XRD analysis begin   ###
###   setup for graph   ###
figsize=xlength*6,ylength*6
dpi = 64.0
fig = plt.figure(figsize=figsize, dpi=dpi)

###   output file   ###
f1=open('result_'+com+'.dat','w')
f1.write('X Y FWHM type peakposition COF peakN FWHM_angle peakposition_angle'\
         +' peak2intensity peak2FWHM'+'\n')
f1.write('\n')

###   predefine params   ###
cat=[[0 for i in range(ylength)] for j in range(xlength)]
F=[[0 for i in range(ylength)] for j in range(xlength)]
E=[[0 for i in range(ylength)] for j in range(xlength)]
M=[[0 for i in range(ylength)] for j in range(xlength)]
h=[[0 for i in range(ylength)] for j in range(xlength)]
COF=[[0 for i in range(ylength)] for j in range(xlength)]
peakN=[[0 for i in range(ylength)] for j in range(xlength)]
F_a=[[0 for i in range(ylength)] for j in range(xlength)]
E_a=[[0 for i in range(ylength)] for j in range(xlength)]
F2=[[0 for i in range(ylength)] for j in range(xlength)]
h2=[[0 for i in range(ylength)] for j in range(xlength)]

###   main loop   ###
n=0
for i in lst:
    x=[]
    try:
        A,B=i.split('.')
        if (B=='data'):
            com,A,B=A.split('_')
            x=int(A)
            y=int(B)
            print(x,y,i)
    except:
        print('Sth. happened!')
        pass

    if x:
        cat[x][y],F[x][y],E[x][y],M[x][y],h[x][y],COF[x][y],peakN[x][y],\
                    F_a[x][y],E_a[x][y]=GetInfo(i,lamda,x,y)
        
    end= datetime.datetime.now()
    print('time'+str(end-start))

for x in range(xlength-1,-1,-1):
    for y in range(ylength-1,-1,-1):
        if (cat[x][y]!=0):
            X=-int(5*(x-10))
            Y=-int(5*(y-10))
            f1.write(str(X)+' '+str(Y)+' '+str(abs(F[x][y]))+' '+str(cat[x][y])\
                     +' '+str(E[x][y])+' '+str(COF[x][y])+' '+str(peakN[x][y])\
                     +' '+str(F_a[x][y])+' '+str(E_a[x][y])+'\n')
    
plt.subplots_adjust(bottom=.01, top=.99, left=.01, right=.99)
plt.savefig('XRD_analysis_'+com+'.png')
plt.close()
f1.close()
print(cat)            
end= datetime.datetime.now()
print('time'+str(end-start))
print('all done!')
